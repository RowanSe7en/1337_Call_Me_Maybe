"""Constrained function-call generation using a language model."""

import json
import math
import numpy as np
from typing import Any, Optional
from pydantic import BaseModel, PrivateAttr, field_validator

_NUM_CHARS: frozenset[str] = frozenset("0123456789.+-eE")
_TERMINATORS: frozenset[str] = frozenset({",", "}"})
_MAX_TOKENS: int = 128


class FunctionCallGenerator(BaseModel):
    """Generate structured function calls using constrained decoding."""

    model: Any
    functions: list[dict[str, Any]]
    prompt: str

    _token_to_id: dict[str, int] = PrivateAttr(default_factory=dict)
    _id_to_token: dict[int, str] = PrivateAttr(default_factory=dict)
    _id_to_decoded: dict[int, str] = PrivateAttr(default_factory=dict)
    _numeric_ids: list[int] = PrivateAttr(default_factory=list)
    _terminator_ids: list[int] = PrivateAttr(default_factory=list)
    _input_ids: list[int] = PrivateAttr(default_factory=list)
    _selected_fn: Optional[dict[str, Any]] = PrivateAttr(default=None)
    _eos_token_id: int = PrivateAttr(default=0)

    @field_validator("functions")
    def functions_not_empty(
        cls,
        v: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Validate that the functions list is not empty."""

        if not v:
            raise ValueError("'functions' must contain at least one entry.")
        return v

    def initialize(self) -> None:
        """Load vocabulary data and prepare token lookup tables."""

        try:
            with open(self._get_vocab_file(), "r") as fh:
                self._token_to_id = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Could not load model vocabulary: {exc}")

        self._id_to_token = {v: k for k, v in self._token_to_id.items()}

        self._id_to_decoded = {
            tid: self._decode([tid]) for tid in self._id_to_token
        }

        self._numeric_ids = [
            tid
            for tid, dec in self._id_to_decoded.items()
            if dec.strip() and all(c in _NUM_CHARS for c in dec.strip())
        ]

        self._terminator_ids = [
            tid
            for tid, dec in self._id_to_decoded.items()
            if dec.strip() in _TERMINATORS
        ]

        if hasattr(self.model, "get_eos_token_id"):
            self._eos_token_id = self.model.get_eos_token_id()
        else:
            self._eos_token_id = self._infer_eos_token_id()

    def _get_vocab_file(self) -> Any:
        """Return the path to the model vocabulary file."""

        return self.model.get_path_to_vocab_file()

    def _infer_eos_token_id(self) -> int:
        """Best-effort EOS lookup for model wrappers that don't expose
        get_eos_token_id() directly (e.g. a black-box SDK like Qwen's
        llm_sdk). Checks the vocab we already loaded for conventional
        end-of-sequence / end-of-turn token strings, in order of
        preference.
        """
        candidates = ("<|endoftext|>", "<|im_end|>", "</s>", "<|eot_id|>")
        for candidate in candidates:
            if candidate in self._token_to_id:
                return self._token_to_id[candidate]

        print(
            "[WARN] Could not infer EOS token id from vocab; "
            "completion-detection on prefix-colliding names may misbehave.",
        )
        return 0

    def _encode(self, text: str) -> list[int]:
        """Encode text into a list of token IDs."""

        result: list[int] = self.model.encode(text).tolist()[0]
        return result

    def _decode(self, ids: list[int]) -> Any:
        """Decode a list of token IDs into text."""

        return self.model.decode(ids)

    def _get_logits(self) -> list[float]:
        """Get the model logits for the current input tokens."""

        result: list[float] = (
            self.model.get_logits_from_input_ids(self._input_ids)
        )
        return result

    def _force(self, text: str) -> None:
        """Append encoded text tokens to the current input."""

        self._input_ids.extend(self._encode(text))

    def _constrained_decoding(
        self, allowed_ids: list[int]
    ) -> tuple[int, str]:
        """Select the highest-scoring token from allowed token IDs."""

        if not allowed_ids:
            raise RuntimeError(
                "Constrained sampling received an empty allowed set."
                )

        logits = self._get_logits()
        masked: list[float] = [-math.inf] * len(logits)
        for tid in allowed_ids:
            if 0 <= tid < len(masked):
                masked[tid] = logits[tid]

        chosen_id = int(np.argmax(masked))
        self._input_ids.append(chosen_id)
        return chosen_id, self._id_to_decoded.get(chosen_id, "")

    def _generate_name(self) -> None:
        """Generate and resolve a function name using constrained decoding."""

        active: dict[str, list[int]] = {
            fn["name"]: self._encode(fn["name"]) for fn in self.functions
        }
        pos: int = 0
        while True:
            completed = [
                name for name, seq in active.items() if pos == len(seq)
            ]
            longer_exist = any(len(seq) > pos for seq in active.values())

            if completed and not longer_exist:
                self._resolve(completed[0])
                return

            allowed: list[int] = [
                seq[pos] for seq in active.values() if len(seq) > pos
            ]

            if completed:
                allowed = allowed + [self._eos_token_id]

            if not allowed:
                raise RuntimeError(
                    f"Name generation stuck at position {pos}: "
                    f"no reachable tokens among {list(active.keys())}"
                )

            chosen_id, _ = self._constrained_decoding(allowed)

            if completed and chosen_id == self._eos_token_id:
                self._resolve(completed[0])
                return

            active = {
                name: seq
                for name, seq in active.items()
                if len(seq) > pos and seq[pos] == chosen_id
            }
            pos += 1

            if not active:
                raise RuntimeError(
                    "No function name could be resolved"
                )

    def _resolve(self, name: str) -> None:
        """Select the function matching the generated name."""

        for f in self.functions:
            if f["name"] == name:
                self._selected_fn = f
                break

    def _generate_number(self, as_float: bool) -> float:
        """Generate a constrained numeric value from model tokens."""

        number_parts: list[str] = []
        has_digit: bool = False
        for _ in range(_MAX_TOKENS):
            current: str = "".join(number_parts)

            allowed: list[int] = []
            for tid in self._numeric_ids:
                stripped = self._id_to_decoded[tid].strip()
                if stripped == "-" and (current or "-" in current):
                    continue
                if stripped == "." and "." in current:
                    continue
                if stripped in ("e", "E") and any(
                    c in current for c in ("e", "E")
                ):
                    continue
                allowed.append(tid)

            if has_digit:
                allowed.extend(self._terminator_ids)

            if not allowed:
                raise RuntimeError(
                    f"Number generation stuck after emitting: {current}"
                )

            chosen_id, tok_str = self._constrained_decoding(allowed)

            stripped_tok = tok_str.strip()

            if stripped_tok in _TERMINATORS:
                self._input_ids.pop()
                break

            if not has_digit:
                if any(c.isdigit() for c in stripped_tok):
                    has_digit = True

            number_parts.append(stripped_tok)

        number_str = "".join(number_parts) or "0"

        if (
            as_float
            and "." not in number_str
            and "e" not in number_str.lower()
        ):
            number_str += ".0"

        return float(number_str)

    def _generate_string(self) -> str:
        """Generate a string value using constrained token decoding."""

        self._force('"')

        value_parts: list[str] = []
        first_token: bool = True

        for _ in range(_MAX_TOKENS):
            logits = self._get_logits()
            chosen_id = int(np.argmax(logits))
            tok_decoded = self._id_to_decoded.get(chosen_id, "")

            if first_token:
                tok_decoded = tok_decoded.lstrip()
                first_token = False

            if '"' in tok_decoded:
                before = tok_decoded.split('"')[0]
                if before:
                    self._force(before)
                    value_parts.append(before)
                self._force('"')
                return "".join(value_parts)

            self._input_ids.append(chosen_id)
            value_parts.append(tok_decoded)

        self._force('"')

        return "".join(value_parts)

    def _generate_boolean(self) -> bool:
        """Generate a boolean value by comparing
        true and false token scores."""

        true_ids: list[int] = self._encode("true")
        false_ids: list[int] = self._encode("false")

        logits = self._get_logits()
        true_score = (
            logits[true_ids[0]]
            if true_ids and true_ids[0] < len(logits)
            else -math.inf
        )
        false_score = (
            logits[false_ids[0]]
            if false_ids and false_ids[0] < len(logits)
            else -math.inf
        )

        if true_score >= false_score:
            self._force("true")
            return True
        else:
            self._force("false")
            return False

    def generate(self) -> dict[str, Any]:
        """Generate a function call with
        extracted parameters from the prompt."""

        if not self._token_to_id:
            self.initialize()

        fn_descriptions: str = "\n".join(
            f"- {fn['name']}: {fn.get('description', '')}"
            for fn in self.functions
        )
        instruction: str = (
            "You are a function-calling assistant.\n"
            f"Available functions:\n{fn_descriptions}\n\n"
            f'User request: "{self.prompt}"\n\n'
            "Select the most appropriate function and extract its arguments.\n"
            "Output JSON:"
        )

        self._input_ids = self._encode(instruction)
        self._selected_fn = None

        self._force('{"prompt":"')

        self._input_ids.extend(self._encode(self.prompt))

        self._force('","name":"')

        self._generate_name()

        if self._selected_fn is None:
            raise ValueError(
                "No function was selected during name generation."
            )

        self._force('","parameters":{')

        params: dict[str, Any] = self._selected_fn.get("parameters", {})
        parameters: dict[str, Any] = {}

        for i, (pname, pschema) in enumerate(params.items()):
            ptype: str = pschema.get("type", "string")

            is_last: bool = i == len(params) - 1

            self._input_ids.extend(self._encode(f'"{pname}":'))

            if ptype in ("number", "float"):
                val: Any = self._generate_number(as_float=True)
                parameters[pname] = val
            elif ptype == "integer":
                val = int(self._generate_number(as_float=False))
                parameters[pname] = val
            elif ptype == "boolean":
                val = self._generate_boolean()
                parameters[pname] = val
            else:
                val = self._generate_string()
                parameters[pname] = val

            if not is_last:
                self._force(",")

        self._force("}}")

        return {
            "prompt": self.prompt,
            "name": self._selected_fn["name"],
            "parameters": parameters,
        }
