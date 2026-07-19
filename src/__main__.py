import argparse
import sys
import time
from typing import Any

from llm_sdk import Small_LLM_Model

from .display import print_error, print_logo, print_result, print_summary
from .generator import FunctionCallGenerator
from .io_handler import load_and_validate, write_results

parsed = argparse.ArgumentParser()

parsed.add_argument(
    "--functions_definition",
    default="data/input/functions_definition.json"
)

parsed.add_argument(
    "--input",
    default="data/input/function_calling_tests.json"
)

parsed.add_argument(
    "--output",
    default="data/output/function_calling_results.json"
)

def main() -> None:

    args = parsed.parse_args()

    try:
        data = load_and_validate(
            func_file=args.functions_definition,
            prompt_file=args.input,
            output_path=args.output,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    print("Loading model — this may take a moment on first run …")
    try:
        model: Small_LLM_Model = Small_LLM_Model()
    except Exception as exc:
        print(f"[ERROR] Failed to load model: {exc}", file=sys.stderr)
        sys.exit(1)

    print_logo()

    results: list[dict[str, Any]] = []
    total_time: float = 0.0

    for prompt in data.prompts:
        t_start = time.perf_counter()

        try:
            generator = FunctionCallGenerator(
                model=model,
                functions=data.functions,
                prompt=prompt,
            )
            result = generator.generate()
            elapsed = time.perf_counter() - t_start
            total_time += elapsed
            results.append(result)
            print_result(result, elapsed, total_time)

        except KeyboardInterrupt:
            print("\n[Interrupted] Saving partial results …")
            break
        except Exception as exc:
            elapsed = time.perf_counter() - t_start
            total_time += elapsed
            print_error(prompt, str(exc))

    try:
        write_results(results, args.output)
        print(f"\nResults written to: {args.output}")
    except OSError as exc:
        print(f"[ERROR] Could not write output file: {exc}", file=sys.stderr)

    print_summary(
        total_prompts=len(data.prompts),
        successes=len(results),
        total_time=total_time,
    )


main()
