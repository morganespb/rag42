*This activity has been created as part of the 42 curriculum by <login>.*

# RAG against the machine

## Description

This project is a small Retrieval Augmented Generation (RAG) system that
answers questions about the vLLM codebase. It cuts the vLLM files into
chunks, finds the chunks that best match a question with BM25, and gives
them to the `Qwen/Qwen3-0.6B` model so it can write an answer based on
real source code and documentation.

## Instructions

Requirements: Python 3.10+, [uv](https://docs.astral.sh/uv/), and a few GB
of free disk space for PyTorch and the model.

```bash
make install            # uv sync
# put the vLLM sources in data/raw/vllm-0.10.1/
# put the datasets in data/datasets/{AnsweredQuestions,UnansweredQuestions}/
make run                # builds the index
make lint               # flake8 + mypy
make clean              # removes caches
```

## Example usage

```bash
# 1. Build the index (about 10 seconds)
uv run python -m src index --max_chunk_size 2000

# 2. Search one question
uv run python -m src search "How do I load a LoRA adapter?" --k 5

# 3. Search a whole dataset
uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --k 10 --save_directory data/output/search_results/UnansweredQuestions

# 4. Check our own recall
uv run python -m src evaluate \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json

# 5. Answer one question, or a whole search results file
uv run python -m src answer "How do I load a LoRA adapter?" --k 3
uv run python -m src answer_dataset \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

## System architecture

```
data/raw/  --> chunking.py --> indexer.py --> data/processed/index.pkl
                                                   |
question  ------------------------------> retriever.py --> top-k sources
                                                   |
                                     generator.py (Qwen3-0.6B) --> answer
```

| File | Role |
| --- | --- |
| `src/__main__.py` | Fire CLI, one method per command, error handling |
| `src/models.py` | Pydantic models shared by all stages |
| `src/chunking.py` | The two chunking strategies (Python and Markdown) |
| `src/indexer.py` | Tokenizer, BM25 index, save and load |
| `src/retriever.py` | Search, dataset search, recall@k |
| `src/generator.py` | Prompt building and answer generation with Qwen |

## Chunking strategy

Both strategies cut the file into blocks at natural boundaries, then glue
small neighbouring blocks together as long as they stay under
`max_chunk_size` (2000 by default). Blocks that are still too big are cut at
line ends.

- **Python**: a new block starts at each `def`, `async def`, `class` or
  decorator line, so a function stays in one piece when it fits.
- **Markdown / text**: a new block starts at each title (`#`, `##`, ...).
  Long sections are cut at blank lines (paragraphs).

Every chunk keeps its file path and its start and end character positions.

## Retrieval method

BM25, written by hand in `indexer.py` with an inverted index (for each
word, the list of chunks that contain it and how many times).

- Score = sum over the question's words of `idf(word) * tf_part`, with the
  usual `k1 = 1.5` and `b = 0.75`.
- The tokenizer keeps identifiers whole and also splits them:
  `get_model_config` gives `get_model_config get model config`, and
  `LoRARequest` gives `lorarequest lo ra request`. This lets a question in
  plain English match code identifiers.
- Common words (`the`, `how`, `vllm`, ...) are ignored.
- The file path is added to the chunk text before indexing, because folder
  and file names often carry the topic (`lora`, `quantization`...).

## Performance analysis

Measured with the moulinette on the public datasets (`--k 10`,
`--max_chunk_size 2000`):

| Dataset | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
| --- | --- | --- | --- | --- |
| Docs (100 questions) | 0.66 | 0.82 | 0.86 | 0.89 |
| Code (99 questions) | 0.55 | 0.73 | 0.80 | 0.87 |

- Indexing: about 10 seconds for 13,000 chunks (limit: 5 minutes).
- Searching 100 questions: about 1 second (limit: 90 seconds for 200).

Removing the stop words raised docs recall@5 from 0.83 to 0.86 and made
search about 4 times faster.

## Design decisions

- BM25 written by hand instead of a library, so every line can be explained.
- Only `.py`, `.md` and `.txt` files are indexed: all the questions point to
  these types.
- The index is one pickle file: simple and fast to load.
- Qwen runs with `transformers`, in non-thinking mode, with a prompt that
  asks to answer only from the context. The context is capped at 6000
  characters so it fits the model's budget.
- `answer_dataset` reads the chunk text back from the files using the
  saved character positions.

## Challenges faced

- Questions use plain words while code uses identifiers: solved by
  splitting `snake_case` and `CamelCase` in the tokenizer.
- The moulinette refuses sources over 2000 characters: the chunker cuts at
  line ends and never goes above `max_chunk_size`, and the CLI refuses a
  value above 2000.
- Bad inputs must never crash: every command runs inside one `try/except`
  that prints a clear error message.

## Resources

- [BM25 on Wikipedia](https://en.wikipedia.org/wiki/Okapi_BM25)
- [Retrieval-Augmented Generation paper (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Python Fire](https://github.com/google/python-fire), [pydantic](https://docs.pydantic.dev/), [uv](https://docs.astral.sh/uv/)

**AI usage**: AI (Claude) was used to explain the subject, propose a file
structure and write a first version of the code. All code was then read,
tested and understood before submission.
