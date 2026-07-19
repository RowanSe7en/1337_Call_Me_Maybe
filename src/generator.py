import json
import math
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, Field, PrivateAttr, field_validator

_NUM_CHARS: frozenset[str] = frozenset("0123456789.+-eE")
_TERMINATORS: frozenset[str] = frozenset({",", "}"})
_MAX_TOKENS: int = 128  # safety cap for any single generation loop

class FunctionCallGenerator(BaseModel):

    model: Any
    functions: list[dict[str, Any]]
    prompt: str

    _token_to_id: dict[str, int] = PrivateAttr(default_factory=dict)
    _id_to_token: dict[int, str] = PrivateAttr(default_factory=dict)
    _id_to_decoded: dict[int, str] = PrivateAttr(default_factory=dict)
    _numeric_ids: list[int] = PrivateAttr(default_factory=list)
    _terminator_ids: list[int] = PrivateAttr(default_factory=list)
    _input_ids: list[int] = PrivateAttr(default_factory=list) #done
    _selected_fn: Optional[dict[str, Any]] = PrivateAttr(default=None) #done

    @field_validator("functions")
    @classmethod
    def functions_not_empty(cls, v: list[dict[str, Any]]) -> list[dict[str, Any]]:

        if not v:
            raise ValueError("'functions' must contain at least one entry.")
        return v

    def initialize(self) -> None:

        try:
            with open(self.model.get_path_to_vocab_file(), "r") as fh:
                self._token_to_id = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Could not load model vocabulary: {exc}")

        self._id_to_token = {v: k for k, v in self._token_to_id.items()}

        self._id_to_decoded = {
            tid: self.model.decode([tid]) for tid in self._id_to_token
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

    def _encode(self, text: str) -> list[int]:

        result: list[int] = self.model.encode(text).tolist()[0]
        return result

    def _get_logits(self) -> list[float]:

        result: list[float] = self.model.get_logits_from_input_ids(self._input_ids)
        return result

    def _force(self, text: str) -> None:

        self._input_ids.extend(self._encode(text))

    def _sample_constrained(
        self, allowed_ids: list[int]
    ) -> tuple[int, str]:

        if not allowed_ids:
            raise RuntimeError("Constrained sampling received an empty allowed set.")

        logits = self._get_logits()
        masked: list[float] = [-math.inf] * len(logits)
        for tid in allowed_ids:
            if 0 <= tid < len(masked):
                masked[tid] = logits[tid]

        chosen_id = int(np.argmax(masked))
        self._input_ids.append(chosen_id)
        return chosen_id, self._id_to_decoded.get(chosen_id, "")

    def _generate_name(self) -> None:

        active: dict[str, list[int]] = {
            fn["name"]: self._encode(fn["name"]) for fn in self.functions
        }
        pos: int = 0
        while True:
            completed = [name for name, seq in active.items() if pos == len(seq)]
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

            chosen_id, _ = self._sample_constrained(allowed)

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
                    "No function name could be resolved — active set exhausted."
                )

    def _resolve(self, name: str) -> None:
        for f in self.functions:
            if f["name"] == name:
                self._selected_fn = f
                break

    def _generate_number(self, as_float: bool) -> float:
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
                    f"Number generation stuck after emitting: {current!r}"
                )

            chosen_id, tok_str = self._sample_constrained(allowed)

            stripped_tok = tok_str.strip()

            if stripped_tok in _TERMINATORS:
                self._input_ids.pop()
                break

            if not has_digit:
                if any(c.isdigit() for c in stripped_tok):
                    has_digit = True

            number_parts.append(stripped_tok)

        number_str = "".join(number_parts) or "0"

        if as_float and "." not in number_str and "e" not in number_str.lower():
            number_str += ".0"

        return float(number_str)

    def _generate_string(self) -> str:
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

        # Safety fallback: close the string without crashing
        self._force('"')
        return "".join(value_parts)

    # -----------------------------------------------------------------------
    # Boolean generation
    # -----------------------------------------------------------------------

    def _generate_boolean(self) -> bool:
        """Generate a JSON boolean value (``true`` or ``false``).

        The logit scores for the first token of ``"true"`` and ``"false"`` are
        compared.  The option with the higher score has its full token sequence
        forced into the model context.

        Returns:
            ``True`` or ``False`` according to the model's preference.
        """
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

    # -----------------------------------------------------------------------
    # Public entry point
    # -----------------------------------------------------------------------

    def generate(self) -> dict[str, Any]:
        """Run the full constrained generation pipeline for one prompt.

        Builds an instruction context, then forces the JSON skeleton while
        using constrained decoding for all dynamic values.  The result is a
        Python dict that can be directly serialised to the output JSON.

        Returns:
            A dict with keys ``"prompt"``, ``"name"``, and ``"parameters"``.

        Raises:
            RuntimeError: If generation fails at any stage (e.g. name not
                found, number generation stuck).
        """
        if not self._token_to_id:
            self.initialize()
        # Build instruction context so the model understands the task
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

        # ── Force opening structure + prompt value ──────────────────────────
        self._force('{"prompt":"')

        # Append prompt tokens to model context; use raw prompt in JSON output
        self._input_ids.extend(self._encode(self.prompt))

        # ── Force name key prefix ────────────────────────────────────────────
        self._force('","name":"')

        # ── Constrained name selection ───────────────────────────────────────
        self._generate_name()

        if self._selected_fn is None:
            raise ValueError("No function was selected during name generation.")

        # ── Force parameters prefix ──────────────────────────────────────────
        self._force('","parameters":{')

        params: dict[str, Any] = self._selected_fn.get("parameters", {})
        parameters: dict[str, Any] = {}

        for i, (pname, pschema) in enumerate(params.items()):
            ptype: str = pschema.get("type", "string")

            is_last: bool = i == len(params) - 1

            # Force parameter key
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
            else:  # string (default)
                val = self._generate_string()
                parameters[pname] = val

            if not is_last:
                self._force(",")


        # ── Force closing braces ─────────────────────────────────────────────
        self._force("}}")


        return {
            "prompt": self.prompt,
            "name": self._selected_fn["name"],
            "parameters": parameters,
        }