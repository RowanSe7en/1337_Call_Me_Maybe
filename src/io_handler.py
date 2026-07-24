"""Input/output validation and JSON file handling."""

import json
import os
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field, model_validator


ALLOWED_TYPES: frozenset[str] = frozenset(
    {"string", "integer", "boolean", "float", "number"}
)


class TypeSchema(BaseModel):
    """Schema describing a supported parameter or return type."""

    type: str = Field(..., min_length=5)

    @model_validator(mode="after")
    def check_type(self) -> "TypeSchema":
        """Validate that the declared type is supported.

        Returns:
            The validated instance.

        Raises:
            ValueError: If the type is not supported.
        """
        if self.type.strip() not in ALLOWED_TYPES:
            raise ValueError(
                f"Unknown return type {self.type}. "
                f"Allowed: {sorted(ALLOWED_TYPES)}"
            )
        return self


class FunctionDefinition(BaseModel):
    """Schema describing a callable function."""

    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    parameters: dict[str, TypeSchema]
    returns: TypeSchema

    @model_validator(mode="after")
    def check_non_empty(self) -> "FunctionDefinition":
        """Validate that required text fields are not blank.

        Returns:
            The validated instance.

        Raises:
            ValueError: If the name or description is blank.
        """
        if not self.name.strip():
            raise ValueError("Function name must not be blank.")
        if not self.description.strip():
            raise ValueError("Function description must not be blank.")
        return self


class PromptEntry(BaseModel):
    """Schema describing a single prompt."""

    prompt: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def check_non_empty(self) -> "PromptEntry":
        """Validate that the prompt is not blank.

        Returns:
            The validated instance.

        Raises:
            ValueError: If the prompt is blank.
        """
        if not self.prompt.strip():
            raise ValueError("Prompt must not be blank.")
        return self


class LoadedData(BaseModel):
    """Validated function definitions and prompts."""

    functions: list[dict[str, Any]]
    prompts: list[str]


def load_and_validate(
    func_file: str,
    prompt_file: str,
    output_path: str,
) -> LoadedData:
    """Load and validate the project input files.

    Args:
        func_file: Path to the function definition JSON file.
        prompt_file: Path to the prompt JSON file.
        output_path: Path where the output file will be written.

    Returns:
        A validated collection of functions and prompts.

    Raises:
        FileNotFoundError: If an input file does not exist.
        ValueError: If the input JSON is invalid.
    """
    func_path = Path(func_file)
    prt_path = Path(prompt_file)

    if not func_path.exists() or not prt_path.exists():
        raise FileNotFoundError("Input file not found")

    with open(func_path, "r") as my_json:
        raw_funcs: Any = json.load(my_json)

    if not raw_funcs or not isinstance(raw_funcs, list):
        raise ValueError(
            f"{func_file}: expected a non-empty JSON array of "
            "function definitions."
        )

    validated_funcs: list[dict[str, Any]] = []

    for idx, entry in enumerate(raw_funcs, start=1):
        if not isinstance(entry, dict):
            raise ValueError(
                f"{func_file}: entry #{idx} must be a JSON object."
            )

        try:
            fd = FunctionDefinition(**entry)
        except Exception as exc:
            raise ValueError(
                f"{func_file}: function #{idx} failed validation — {exc}"
            )

        validated_funcs.append(fd.model_dump())

    with open(prt_path, "r") as my_json:
        raw_prompts: Any = json.load(my_json)

    if not raw_prompts or not isinstance(raw_prompts, list):
        raise ValueError(
            f"{prompt_file}: expected a non-empty JSON array of "
            "prompt objects."
        )

    validated_prompts: list[str] = []

    for idx, entry in enumerate(raw_prompts, start=1):
        if not isinstance(entry, dict):
            raise ValueError(
                f"{prompt_file}: entry #{idx} must be a JSON object."
            )

        try:
            pe = PromptEntry(**entry)
        except Exception as exc:
            raise ValueError(
                f"{prompt_file}: prompt #{idx} failed validation — {exc}"
            )

        validated_prompts.append(pe.prompt)

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)

    return LoadedData(
        functions=validated_funcs,
        prompts=validated_prompts,
    )


def write_results(results: list[dict[str, Any]], output_path: str) -> None:
    """Write generated results to a JSON file.

    Args:
        results: Results produced by the generator.
        output_path: Destination JSON file.
    """
    with open(output_path, "w") as my_json:
        json.dump(results, my_json, indent=2)
