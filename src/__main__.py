import argparse
import sys

from .display import print_logo
from .io_handler import load_and_validate

parsed = argparse.ArgumentParser()

parsed.add_argument(
    "--functions_definition",
    default="data/input/functions_definition.json"
)

parsed.add_argument(
    "--input",
    default="data/input/function_calling_tests.json"
)


def main() -> None:
    args = parsed.parse_args()

    print_logo()

    try:
        load_and_validate(
            func_file=args.functions_definition,
            prompt_file=args.input,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


main()
