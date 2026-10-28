"""Command-line interface: uv run python -m src <command> [options]."""

import os
import sys
import time
from typing import List

import fire
from pydantic import ValidationError
from tqdm import tqdm

from src.indexer import build_index, load_index, save_index
from src.models import (MinimalAnswer, RagDataset, StudentSearchResults,
                        StudentSearchResultsAndAnswer)
from src.retriever import (recall_at_k, search, search_dataset,
                           source_to_chunk)

MAX_ALLOWED_CHUNK_SIZE = 2000


def read_text(path: str) -> str:
    """Read a whole text file.

    Args:
        path: The file to read.

    Returns:
        The file content.
    """
    with open(path, encoding="utf-8") as file:
        return file.read()


def write_json(save_directory: str, input_path: str, content: str) -> str:
    """Write JSON into save_directory, named like the input file.

    Args:
        save_directory: The output folder.
        input_path: The input file, whose name is reused.
        content: The JSON text.

    Returns:
        The path of the written file.
    """
    os.makedirs(save_directory, exist_ok=True)
    path = os.path.join(save_directory, os.path.basename(input_path))
    with open(path, "w", encoding="utf-8") as file:
        file.write(content)
    return path


def check_k(k: int) -> None:
    """Refuse a k that is not a positive integer.

    Args:
        k: The number of results asked for.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError(f"k must be a positive integer, got {k!r}")


class RagCLI:
    """Answer questions about the vLLM codebase."""

    def __init__(self, processed_dir: str = "data/processed") -> None:
        """Set where the index is stored.

        Args:
            processed_dir: The folder holding the index.
        """
        self.processed_dir = processed_dir

    def index(self, max_chunk_size: int = 2000,
              raw_dir: str = "data/raw") -> None:
        """Chunk every file under raw_dir and save the index.

        Args:
            max_chunk_size: The maximum number of characters per chunk.
            raw_dir: The folder holding the corpus.
        """
        if not isinstance(max_chunk_size, int) or not (
                0 < max_chunk_size <= MAX_ALLOWED_CHUNK_SIZE):
            raise ValueError("max_chunk_size must be an integer "
                             f"between 1 and {MAX_ALLOWED_CHUNK_SIZE}")
        if not os.path.isdir(raw_dir):
            raise FileNotFoundError(f"folder not found: {raw_dir}")
        start = time.time()
        index = build_index(raw_dir, max_chunk_size)
        save_index(index, self.processed_dir)
        print(f"Indexed {len(index.chunks)} chunks "
              f"in {time.time() - start:.1f}s")
        print(f"Ingestion complete! Indices saved under "
              f"{self.processed_dir}/")

    def search(self, query: str, k: int = 10) -> None:
        """Print the top-k sources for one question.

        Args:
            query: The question.
            k: The number of sources to return.
        """
        check_k(k)
        query = str(query)
        if not query.strip():
            raise ValueError("the query is empty")
        index = load_index(self.processed_dir)
        sources = search(index, query, k)
        if not sources:
            print("No result found.")
        for rank, source in enumerate(sources, start=1):
            print(f"{rank}. {source.file_path} "
                  f"[{source.first_character_index}:"
                  f"{source.last_character_index}]")

    def search_dataset(self, dataset_path: str, k: int = 10,
                       save_directory: str = "data/output/search_results"
                       ) -> None:
        """Search every question of a dataset and save the results.

        Args:
            dataset_path: The JSON dataset of questions.
            k: The number of sources per question.
            save_directory: Where to write the results.
        """
        check_k(k)
        dataset = RagDataset.model_validate_json(read_text(dataset_path))
        index = load_index(self.processed_dir)
        results = search_dataset(index, dataset, k)
        path = write_json(save_directory, dataset_path,
                          results.model_dump_json(indent=2))
        print(f"Saved student_search_results to {path}")

    def answer(self, query: str, k: int = 5) -> None:
        """Answer one question using the retrieved chunks.

        Args:
            query: The question.
            k: The number of chunks given to the model.
        """
        from src.generator import Generator  # slow import, only when needed

        check_k(k)
        query = str(query)
        if not query.strip():
            raise ValueError("the query is empty")
        index = load_index(self.processed_dir)
        chunks = index.search(query, k)
        generator = Generator()
        print(generator.answer(query, chunks))
        print("\nSources:")
        for chunk in chunks:
            print(f"- {chunk.file_path} [{chunk.first_character_index}:"
                  f"{chunk.last_character_index}]")

    def answer_dataset(self, student_search_results_path: str,
                       save_directory: str = (
                           "data/output/search_results_and_answer")
                       ) -> None:
        """Answer every question of a search results file.

        Args:
            student_search_results_path: The output of search_dataset.
            save_directory: Where to write the answers.
        """
        from src.generator import Generator  # slow import, only when needed

        results = StudentSearchResults.model_validate_json(
            read_text(student_search_results_path))
        print(f"Loaded {len(results.search_results)} questions")
        generator = Generator()
        answers: List[MinimalAnswer] = []
        for result in tqdm(results.search_results, desc="Answering"):
            chunks = [source_to_chunk(s) for s in result.retrieved_sources]
            answers.append(MinimalAnswer(
                question_id=result.question_id,
                question=result.question,
                retrieved_sources=result.retrieved_sources,
                answer=generator.answer(result.question, chunks)))
        output = StudentSearchResultsAndAnswer(search_results=answers,
                                               k=results.k)
        path = write_json(save_directory, student_search_results_path,
                          output.model_dump_json(indent=2))
        print(f"Saved student_search_results_and_answer to {path}")

    def evaluate(self, student_search_results_path: str,
                 dataset_path: str) -> None:
        """Print our own recall@k against a dataset with correct sources.

        Args:
            student_search_results_path: The output of search_dataset.
            dataset_path: The AnsweredQuestions dataset.
        """
        results = StudentSearchResults.model_validate_json(
            read_text(student_search_results_path))
        dataset = RagDataset.model_validate_json(read_text(dataset_path))
        line = []
        for k in (1, 3, 5, 10):
            if k <= results.k:
                line.append(f"Recall@{k}: "
                            f"{recall_at_k(results, dataset, k):.3f}")
        print(" ".join(line))


def main() -> None:
    """Run the CLI and turn every error into a clean message."""
    try:
        fire.Fire(RagCLI)
    except FileNotFoundError as error:
        print(f"Error: file not found: {error.filename or error}",
              file=sys.stderr)
        sys.exit(1)
    except ValidationError as error:
        print(f"Error: invalid JSON content:\n{error}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        sys.exit(1)
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
