"""Search the index and measure recall@k."""

from typing import Dict, List

from tqdm import tqdm

from src.indexer import Index
from src.models import (AnsweredQuestion, Chunk, MinimalSearchResults,
                        MinimalSource, RagDataset, StudentSearchResults)


def search(index: Index, question: str, k: int) -> List[MinimalSource]:
    """Find the k best sources for one question.

    Args:
        index: The loaded index.
        question: The question text.
        k: The number of sources to return.

    Returns:
        The best sources, best first.
    """
    chunks = index.search(question, k)
    return [MinimalSource(file_path=c.file_path,
                          first_character_index=c.first_character_index,
                          last_character_index=c.last_character_index)
            for c in chunks]


def search_dataset(index: Index, dataset: RagDataset,
                   k: int) -> StudentSearchResults:
    """Search every question of a dataset.

    Args:
        index: The loaded index.
        dataset: The questions.
        k: The number of sources per question.

    Returns:
        The search results for all questions.
    """
    results: List[MinimalSearchResults] = []
    for question in tqdm(dataset.rag_questions, desc="Searching"):
        results.append(MinimalSearchResults(
            question_id=question.question_id,
            question=question.question,
            retrieved_sources=search(index, question.question, k)))
    return StudentSearchResults(search_results=results, k=k)


def overlaps(found: MinimalSource, truth: MinimalSource) -> bool:
    """Check if a found source matches a correct one (IoU >= 0.05).

    Args:
        found: A source we retrieved.
        truth: A correct source from the dataset.

    Returns:
        True if same file and the ranges overlap enough.
    """
    if found.file_path != truth.file_path:
        return False
    common = (min(found.last_character_index, truth.last_character_index)
              - max(found.first_character_index, truth.first_character_index))
    union = (max(found.last_character_index, truth.last_character_index)
             - min(found.first_character_index, truth.first_character_index))
    return union > 0 and common / union >= 0.05


def recall_at_k(results: StudentSearchResults, dataset: RagDataset,
                k: int) -> float:
    """Compute the average share of correct sources found in the top k.

    Args:
        results: Our search results.
        dataset: The dataset with the correct sources.
        k: How many of our results to look at.

    Returns:
        The recall, between 0 and 1.
    """
    found_by_id: Dict[str, List[MinimalSource]] = {
        r.question_id: r.retrieved_sources[:k] for r in results.search_results}
    scores: List[float] = []
    for question in dataset.rag_questions:
        if not isinstance(question, AnsweredQuestion) or not question.sources:
            continue
        found = found_by_id.get(question.question_id, [])
        hits = sum(1 for truth in question.sources
                   if any(overlaps(f, truth) for f in found))
        scores.append(hits / len(question.sources))
    return sum(scores) / len(scores) if scores else 0.0


def source_to_chunk(source: MinimalSource) -> Chunk:
    """Read the text of a source back from its file.

    Args:
        source: The source to read.

    Returns:
        A chunk holding the source's text.
    """
    with open(source.file_path, encoding="utf-8") as file:
        text = file.read()
    return Chunk(file_path=source.file_path,
                 first_character_index=source.first_character_index,
                 last_character_index=source.last_character_index,
                 text=text[source.first_character_index:
                           source.last_character_index])
