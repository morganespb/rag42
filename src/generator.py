"""Generate an answer with Qwen from the retrieved chunks."""

from typing import Any, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models import Chunk

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
MAX_CONTEXT_CHARS = 6000  # keeps the prompt well inside the model's budget
MAX_NEW_TOKENS = 256


class Generator:
    """Loads the model once and answers questions with it."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer: Any = AutoTokenizer.from_pretrained(model_name)
        model: Any = AutoModelForCausalLM.from_pretrained(model_name)
        self.model: Any = model.to(self.device)

    def build_prompt(self, question: str, chunks: List[Chunk]) -> str:
        """Put the question and the chunks together into a prompt"""

        context = ""
        for chunk in chunks:
            part = f"--- {chunk.file_path}\n{chunk.text}\n"
            if len(context) + len(part) > MAX_CONTEXT_CHARS:
                break
            context += part
        messages = [
            {"role": "system",
             "content": "You answer questions about the vLLM codebase. "
                        "Use only the context given. If the context does "
                        "not contain the answer, say you don't know. "
                        "Answer in 2 to 4 sentences."},
            {"role": "user",
             "content": f"Context:\n{context}\nQuestion: {question}"},
        ]
        prompt: str = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        return prompt

    def answer(self, question: str, chunks: List[Chunk]) -> str:
        """Generate an answer to the question from the chunks"""

        prompt = self.build_prompt(question, chunks)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        text: str = self.tokenizer.decode(new_tokens,
                                          skip_special_tokens=True)
        return text.strip()
