"""Build the search index: read files, chunk them, compute BM25 data."""

import math
import os
import pickle
import re
from collections import Counter
from typing import Dict, List

from tqdm import tqdm

from src.chunking import chunk_file
from src.models import Chunk

EXTENSIONS = (".py", ".md", ".txt")
INDEX_FILE = "index.pkl"

# Very common words that do not help the search.
STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "be", "to", "of", "in", "on",
    "for", "and", "or", "what", "which", "how", "does", "do", "when",
    "where", "why", "with", "by", "it", "its", "that", "this", "as", "can",
    "from", "at", "if", "vllm", "s",
}

# BM25 settings: k1 controls word repetition, b controls length penalty.
K1 = 1.5
B = 0.75


def tokenize(text: str) -> List[str]:
    """Turn a text into a list of lowercase words.

    Identifiers are kept whole and also split into their parts, so
    "get_model_config" gives "get_model_config", "get", "model", "config"
    and "LoRARequest" gives "lorarequest", "lo", "ra", "request".

    Args:
        text: Any text: a question or a chunk of code.

    Returns:
        The list of tokens.
    """
    tokens: List[str] = []
    for word in re.findall(r"[A-Za-z0-9_]+", text):
        parts = [p for p in word.split("_") if p]
        sub_words: List[str] = []
        for part in parts:
            sub_words.extend(
                re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+",
                           part))
        if word.lower() in STOP_WORDS:
            continue
        tokens.append(word.lower())
        if len(sub_words) > 1:
            tokens.extend(w.lower() for w in sub_words)
    return tokens


def find_files(raw_dir: str) -> List[str]:
    """List the files to index under raw_dir.

    Args:
        raw_dir: The folder holding the corpus.

    Returns:
        The sorted list of file paths.
    """
    paths: List[str] = []
    for folder, _, names in os.walk(raw_dir):
        for name in names:
            if name.endswith(EXTENSIONS):
                paths.append(os.path.join(folder, name))
    return sorted(paths)


class Index:
    """The chunks plus everything BM25 needs to score them."""

    def __init__(self) -> None:
        """Create an empty index."""
        self.chunks: List[Chunk] = []
        # word -> list of (chunk number, how many times the word appears)
        self.postings: Dict[str, List[tuple[int, int]]] = {}
        self.idf: Dict[str, float] = {}
        self.lengths: List[int] = []
        self.average_length: float = 0.0

    def add_chunk(self, chunk: Chunk) -> None:
        """Add one chunk and count its words.

        Args:
            chunk: The chunk to add.
        """
        number = len(self.chunks)
        self.chunks.append(chunk)
        # The path is added to the text: its words help a lot.
        tokens = tokenize(chunk.file_path + " " + chunk.text)
        self.lengths.append(len(tokens))
        for word, count in Counter(tokens).items():
            self.postings.setdefault(word, []).append((number, count))

    def finish(self) -> None:
        """Compute the IDF of every word and the average chunk length."""
        total = len(self.chunks)
        for word, posting in self.postings.items():
            n = len(posting)
            self.idf[word] = math.log(1 + (total - n + 0.5) / (n + 0.5))
        if total:
            self.average_length = sum(self.lengths) / total

    def search(self, query: str, k: int) -> List[Chunk]:
        """Return the k chunks with the best BM25 score for the query"""
        scores: Dict[int, float] = {}
        for word in set(tokenize(query)):
            for number, count in self.postings.get(word, []):
                length_ratio = self.lengths[number] / self.average_length
                tf = count * (K1 + 1) / (
                    count + K1 * (1 - B + B * length_ratio))
                scores[number] = scores.get(number, 0.0) + self.idf[word] * tf
        best = sorted(scores, key=lambda n: scores[n], reverse=True)[:k]
        return [self.chunks[n] for n in best]


def build_index(raw_dir: str, max_chunk_size: int) -> Index:
    """Read every file under raw_dir and build the index"""
    index = Index()
    for path in tqdm(find_files(raw_dir), desc="Indexing"):
        try:
            with open(path, encoding="utf-8") as file:
                text = file.read()
        except (OSError, UnicodeDecodeError):
            continue  # skip files we cannot read
        for start, end in chunk_file(path, text, max_chunk_size):
            index.add_chunk(Chunk(file_path=path,
                                  first_character_index=start,
                                  last_character_index=end,
                                  text=text[start:end]))
    index.finish()
    return index


def save_index(index: Index, processed_dir: str) -> str:
    """Save the index to processed_dir"""

    os.makedirs(processed_dir, exist_ok=True)
    path = os.path.join(processed_dir, INDEX_FILE)
    with open(path, "wb") as file:
        pickle.dump(index, file)
    return path


def load_index(processed_dir: str) -> Index:
    """Load the index saved by save_index"""

    path = os.path.join(processed_dir, INDEX_FILE)
    with open(path, "rb") as file:
        index: Index = pickle.load(file)
    return index
