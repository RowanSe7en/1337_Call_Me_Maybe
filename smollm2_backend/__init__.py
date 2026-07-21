"""Thin adapter exposing SmolLM2-1.7B-Instruct through the same
interface FunctionCallGenerator already expects from `self.model`:

    get_path_to_vocab_file() -> str
    encode(text: str)        -> tensor-like, supports .tolist()[0]
    decode([token_id])       -> str
    get_logits_from_input_ids(ids: list[int]) -> list[float]
    get_eos_token_id()       -> int

No changes to generator.py's algorithm are required — only the small
_eos_token_id wiring noted in initialize() (see generator.py patch).
"""

import json
import os
import tempfile
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_MODEL_NAME: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct"


class SmolLM2Backend:
    """Adapter around SmolLM2-1.7B-Instruct for constrained decoding."""

    def __init__(self, model_name: str = _MODEL_NAME) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        ).to(self.device)
        self.model.eval()

        # generator.py expects a plain token_str -> id JSON file on disk.
        # Build it directly from the tokenizer's own vocab rather than
        # relying on save_vocabulary(), which isn't guaranteed to exist
        # in the same shape across tokenizer backends.
        vocab: dict[str, int] = self.tokenizer.get_vocab()
        tmp_dir = tempfile.mkdtemp(prefix="smollm2_vocab_")
        self._vocab_path = os.path.join(tmp_dir, "vocab.json")
        with open(self._vocab_path, "w") as fh:
            json.dump(vocab, fh)

    def get_path_to_vocab_file(self) -> str:
        return self._vocab_path

    def encode(self, text: str) -> Any:
        # add_special_tokens=False: generator.py calls this repeatedly on
        # small text fragments while building up a forced context — we
        # don't want a BOS/EOS re-inserted mid-sequence each time.
        return self.tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        ).input_ids

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        ids_tensor = torch.tensor([input_ids], device=self.device)
        with torch.no_grad():
            output = self.model(ids_tensor)
        # last position's logits over the full vocab
        return output.logits[0, -1, :].tolist()

    def get_eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id
