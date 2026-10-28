"""Cut files into chunks of at most max_chunk_size characters.
1. cut the text into "blocks" at natural boundaries
   (titles for Markdown, def/class lines for Python).
2. glue small blocks back together while they fit in max_chunk_size.
3. cut any block that is still too big into smaller slices.
"""

import re
from typing import List, Tuple

Span = Tuple[int, int]

PYTHON_BOUNDARY = re.compile(r"^\s*(@|def |async def |class )")
MARKDOWN_BOUNDARY = re.compile(r"^#{1,6} ")


def split_lines(text: str) -> List[Span]:
    """Return the span of every line, newline included"""
    spans: List[Span] = []
    start = 0
    for line in text.splitlines(keepends=True):
        spans.append((start, start + len(line)))
        start += len(line)
    return spans


def split_at_boundaries(text: str, boundary: re.Pattern[str]) -> List[Span]:
    """Cut the text into blocks, starting a new block at each boundary line.

    Args:
        text: The file content.
        boundary: A regex matching the lines that start a new block.

    Returns:
        A list of (start, end) spans covering the whole text.
    """
    blocks: List[Span] = []
    block_start = 0
    previous_was_decorator = False
    for start, end in split_lines(text):
        line = text[start:end]
        is_boundary = boundary.match(line) is not None
        # Keep a decorator (@...) together with the def under it.
        if is_boundary and start > block_start and not previous_was_decorator:
            blocks.append((block_start, start))
            block_start = start
        previous_was_decorator = line.lstrip().startswith("@")
    if block_start < len(text):
        blocks.append((block_start, len(text)))
    return blocks


def cut_big_block(text: str, span: Span, max_size: int) -> List[Span]:
    """Cut a block that is too big into slices, at line ends if possible.
    Returns: a list of spans, each at most max_size characters long"""
    start, end = span
    slices: List[Span] = []
    while end - start > max_size:
        cut = text.rfind("\n", start, start + max_size)
        if cut <= start:
            cut = start + max_size  # one very long line: cut it anywhere
        else:
            cut += 1  # keep the newline in the first slice
        slices.append((start, cut))
        start = cut
    if start < end:
        slices.append((start, end))
    return slices


def merge_blocks(text: str, blocks: List[Span], max_size: int) -> List[Span]:
    """Glue consecutive small blocks together while they fit in max_size.

    Args:
        text: The file content.
        blocks: The spans to merge, in order.
        max_size: The maximum number of characters per chunk.

    Returns:
        The final list of chunk spans.
    """
    small_blocks: List[Span] = []
    for block in blocks:
        small_blocks.extend(cut_big_block(text, block, max_size))

    chunks: List[Span] = []
    for start, end in small_blocks:
        if chunks and end - chunks[-1][0] <= max_size:
            chunks[-1] = (chunks[-1][0], end)
        else:
            chunks.append((start, end))
    return [(s, e) for s, e in chunks if text[s:e].strip()]


def chunk_python(text: str, max_size: int) -> List[Span]:
    """Chunk Python code, cutting before def/class lines.

    Args:
        text: The Python file content.
        max_size: The maximum number of characters per chunk.

    Returns:
        A list of chunk spans.
    """
    blocks = split_at_boundaries(text, PYTHON_BOUNDARY)
    return merge_blocks(text, blocks, max_size)


def chunk_markdown(text: str, max_size: int) -> List[Span]:
    """Chunk Markdown or plain text, cutting before titles and blank lines"""

    sections = split_at_boundaries(text, MARKDOWN_BOUNDARY)
    blocks: List[Span] = []
    for start, end in sections:
        if end - start <= max_size:
            blocks.append((start, end))
            continue
        # Section too big: cut it at paragraphs (blank lines) first
        paragraph_start = start
        for match in re.finditer(r"\n\s*\n", text[start:end]):
            cut = start + match.end()
            blocks.append((paragraph_start, cut))
            paragraph_start = cut
        if paragraph_start < end:
            blocks.append((paragraph_start, end))
    return merge_blocks(text, blocks, max_size)


def chunk_file(file_path: str, text: str, max_size: int) -> List[Span]:
    """Pick the right chunking strategy for a file.
    Returns: a list of chunk spans"""

    if file_path.endswith(".py"):
        return chunk_python(text, max_size)
    return chunk_markdown(text, max_size)
