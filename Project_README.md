*This project has been created as part of the 42 curriculum by brouane.*

## Description

`call-me-maybe` is a function-calling engine for LLMs. Given a natural-language
prompt and a list of available functions (name, description, typed parameters),
it produces the correct function call as JSON:

```
"What is the sum of 40 and 2?"  ->  {"name": "fn_add_numbers", "parameters": {"a": 40.0, "b": 2.0}}
```

The twist is that the model behind this — **Qwen/Qwen3-0.6B** — is a 500M
parameter model, far too small and too undertrained to reliably output valid,
schema-compliant JSON on its own. Prompting it and hoping for the best gets
you maybe 30% valid JSON. This project instead drives the model with
**constrained decoding**: at every generation step, the raw model logits are
masked so that only tokens which keep the output both syntactically valid
JSON *and* compliant with the function's schema are allowed to be picked. The
result is 100% parseable, schema-correct JSON on every single run, regardless
of how small or unreliable the underlying model is.

## Instructions

### Requirements

* Python 3.10+
* [`uv`](https://docs.astral.sh/uv/) for dependency management
* The `llm_sdk/` package (provided, copied at the repository root next to `src/`)

### Installation

> **1337/42 users:** Before installing anything, run:
>
> ```bash
> make setup_goinfre
> ```
>
> This configures:
>
> * `UV_CACHE_DIR` → `~/goinfre/call_me_brouane/uv-cache`
> * `HF_HOME` → `~/goinfre/call_me_brouane/huggingface`
> * `UV_PROJECT_ENVIRONMENT` → `~/goinfre/call_me_brouane/.venv`
>
> so that the virtual environment and model caches are stored in `goinfre` instead of your home directory, helping you avoid your home partition quota. The target appends these environment variables to `~/.zshrc` (only if they are not already present) and prints the commands needed to reload your shell. **Follow those instructions before continuing.**

Install the project dependencies:

```bash
make install
```

This installs `numpy`, `pydantic`, `torch`, `transformers`, `huggingface-hub`, and `rich`. On the first run, `transformers` will also download the `Qwen/Qwen3-0.6B` model from the Hugging Face Hub.

### Running

Run the project with:

```bash
make run
```

This is equivalent to:

```bash
uv run python -m src \
  [--functions_definition <path>] \
  [--input <path>] \
  [--output <path>]
```

By default:

| Argument                 | Default                                     |
| ------------------------ | ------------------------------------------- |
| `--functions_definition` | `data/input/functions_definition.json`      |
| `--input`                | `data/input/function_calling_tests.json`    |
| `--output`               | `data/output/function_calling_results.json` |

Example:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```
### Makefile Targets

| Target | Description |
|--------|-------------|
| <code>make&nbsp;setup_goinfre</code> | Configure the `uv` cache, Hugging Face cache, and virtual environment to use `goinfre` (recommended on 1337/42 machines). |
| <code>make&nbsp;install</code> | Install the project dependencies using `uv sync`. |
| <code>make&nbsp;run</code> | Run the project. |
| <code>make&nbsp;debug</code> | Run the project under the Python debugger (`pdb`). |
| <code>make&nbsp;lint</code> | Run `flake8` and `mypy` with the project's linting checks. |
| <code>make&nbsp;clean</code> | Remove Python bytecode (`__pycache__`) and mypy/Ruff caches. |
| <code>make&nbsp;fclean_goinfre</code> | Run `make clean` and remove the entire `goinfre` project directory. |
| <code>make&nbsp;fclean</code> | Run `make clean` and remove local virtual environments and caches created when `setup_goinfre` was not used. |
| <code>make&nbsp;re_goinfre</code> | Recreate the `goinfre` environment by running `make fclean_goinfre` followed by `make install`. |
| <code>make&nbsp;re</code> | Reinstall the project by running `make fclean` followed by `make install`. |

## Folder structure

```
1337_Call_Me_Maybe/
└────── Makefile
    ├── README.md
    ├── data/
    │   └── input/
    │       ├── function_calling_tests.json
    │       └── functions_definition.json
    ├── llm_sdk/
    │   └── __init__.py
    ├── pyproject.toml
    ├── smollm2_backend/
    │   └── __init__.py
    ├── src/
    │   ├── __main__.py
    │   ├── display.py
    │   ├── generator.py
    │   └── io_handler.py
    └── uv.lock
```

## How it works, step by step

### 1. Parsing arguments

`src/__main__.py` builds an `argparse.ArgumentParser` with three optional
flags — `--functions_definition`, `--input`, `--output` — each defaulting to
a path under `data/`. Nothing is required on the command line; the program
runs out of the box on the sample data.

### 2. Loading and validating the input files

`io_handler.load_and_validate()` takes the three raw path strings and does
the actual work:

- Both `func_file` and `prompt_file` are wrapped in `pathlib.Path` so the
  rest of the code can work with proper path objects (`.exists()`, joining,
  etc.) instead of raw strings.
- If either file is missing, it raises `FileNotFoundError` immediately —
  no crash, no traceback dumped on the user, just a clean error caught in
  `main()` and printed to `stderr`.
- Each file is `json.load`-ed inside a `try/except` so malformed JSON (a
  trailing comma, a missing bracket, an empty file) is reported as a clear
  error rather than crashing the program.
- Every entry in `functions_definition.json` is validated against a
  **pydantic** schema (`FunctionDefinition`, itself composed of `TypeSchema`
  for parameter/return types). This guarantees every function has a
  non-empty `name`, a `description`, and parameters whose declared types are
  one of `string`, `integer`, `float`, `number`, `boolean` — before any of
  that data ever reaches the model.
- Every entry in `function_calling_tests.json` is validated the same way
  through a `PromptEntry` model, rejecting blank prompts.
- Finally, the output directory (parent of `--output`) is created with
  `os.makedirs(..., exist_ok=True)` so the program never fails at the very
  last step just because `data/output/` didn't exist yet.

The function returns a single `LoadedData` object holding the validated
`functions` and `prompts`, which is exactly what the rest of the pipeline
consumes.

### 3. Loading the model

`main()` instantiates `Small_LLM_Model` from `llm_sdk`. This wraps
`AutoTokenizer` / `AutoModelForCausalLM` from `transformers`, auto-selects a
device (`mps` > `cuda` > `cpu`), picks an appropriate dtype (`float16` on
GPU/MPS, `float32` on CPU), and puts the model in `eval()` mode with
gradients disabled — this is inference only, never training.

### 4. Building the `FunctionCallGenerator`

For every prompt, `src/__main__.py` creates a fresh
`FunctionCallGenerator(model=model, functions=data.functions, prompt=prompt)`
and calls `.generate()` on it. `FunctionCallGenerator` is itself a pydantic
`BaseModel`, so `model`, `functions`, and `prompt` are validated fields, and
all of the generator's internal bookkeeping lives in private attributes
(`PrivateAttr`) that pydantic doesn't try to validate or serialize:

- `_token_to_id` / `_id_to_token`: the raw vocabulary and its reverse map.
- `_id_to_decoded`: every token ID pre-decoded to its actual text.
- `_numeric_ids` / `_terminator_ids`: token IDs that are safe to use while
  building a number, and tokens that legally end one (`,` or `}`).
- `_input_ids`: the running list of token IDs fed to the model so far.
- `_selected_fn`: the function definition chosen for the current prompt.
- `_eos_token_id`: the model's end-of-sequence token ID, used to disambiguate
  function names where one name is a prefix of another during
  `_generate_name()`.

**`initialize()`** — called once, lazily, the first time `generate()` runs —
does the expensive setup:

1. It downloads/loads `vocab.json` via `model.get_path_to_vocab_file()` and
   parses it into `_token_to_id` (token string → token ID).
2. It builds the reverse mapping, `_id_to_token`.
3. It decodes **every single token ID** once via `model.decode([tid])`,
   caching the result in `_id_to_decoded`. This matters because during
   generation the model only ever produces token *IDs*; having every ID
   pre-decoded means any generated token can be turned into real output text
   instantly, without re-invoking the tokenizer thousands of times.
4. It scans that decoded vocabulary once to precompute `_numeric_ids` (every
   token whose decoded text is made up only of digits and the characters
   `0123456789.+-eE`) and `_terminator_ids` (tokens that decode to `,` or
   `}`). Doing this scan once at startup, instead of on every single
   generated digit, is what keeps number generation fast.
5. It sets `_eos_token_id`: if the model wrapper exposes
   `get_eos_token_id()` directly, that value is used; otherwise
   `_get_eos_token_id()` falls back to scanning the already-loaded vocab
   for conventional end-of-sequence / end-of-turn token strings (in order:
   `<|endoftext|>`, `<|im_end|>`, `</s>`, `<|eot_id|>`), printing a warning
   and defaulting to `0` if none are found.

### 5. The generation pipeline (`generate()`)

`generate()` builds the initial instruction prompt (the list of available
function names + descriptions, plus the user's request), encodes it into
`_input_ids`, and then **forces** a fixed JSON skeleton around the model's
free-form choices. There's no single forcing helper — each literal piece of
scaffolding is pushed onto `_input_ids` inline with
`self._input_ids.extend(self._encode(text))`, which just encodes `text` and
appends the resulting token IDs directly, no sampling involved:

```
{"prompt":"<original prompt>","name":"   <-- forced
```

From there, the model is only ever asked to fill in the parts that actually
require a decision:

**a. Choosing the function name — `_generate_name()`**

This is a trie-style constrained walk. All function names are pre-tokenized
into an `active` dict (`name -> token id sequence`). At each position:

- It keeps only the token IDs that are the *next* token for some
  still-alive candidate name (this is the `allowed` set).
- If any candidate's full token sequence has already been emitted at the
  current position (`completed`), `_eos_token_id` is added to `allowed` too,
  so the model can choose to stop there instead of being forced to keep
  extending into a longer candidate name that happens to share the same
  prefix.
- It calls `_constrained_decoding(allowed)`, which fetches the raw logits via
  `model.get_logits_from_input_ids(self._input_ids)`, sets every logit
  **not** in `allowed` to `-inf`, and takes the `argmax` of what's left —
  so the model can only physically emit a token that keeps at least one
  candidate function name alive (or the EOS token, once a candidate is
  already complete).
- If the chosen token is `_eos_token_id` while a candidate is already
  complete, that candidate is resolved immediately. Otherwise, whichever
  names no longer match the chosen token are dropped from `active`.
- Once exactly one candidate remains and its full token sequence has been
  emitted, `_resolve(name)` looks it up in `self.functions` and stores it in
  `_selected_fn`.

This means the model is choosing *which* function to call token-by-token,
but it is structurally impossible for it to emit a name that isn't one of
the real, provided function names.

**b. Choosing the parameters**

Once a function is selected, `"parameters":{` is forced onto the input, then
for each parameter in the function's schema, its name and `:` are forced,
and the value is generated according to its declared type:

- **`number` / `float` / `integer` — `_generate_number()`**: builds up a
  numeric string one constrained token at a time, using the precomputed
  `_numeric_ids`, with extra rules layered on top (no second `-`, no second
  `.`, no second `e`/`E`) and only allowing a terminator (`,` or `}`) once at
  least one digit has been produced. The accumulated string is cast to
  `float`, and if the schema says `integer` the value is truncated with
  `int(...)`.
- **`string` — `_generate_string()`**: forces an opening `"`, then greedily
  takes the argmax token at each step (unconstrained, since almost any text
  is valid inside a string) until a token containing a closing `"` appears,
  at which point it force-closes the quote and returns the accumulated text.
- **`boolean` — `_generate_boolean()`**: compares the logit for the first
  token of `"true"` against the first token of `"false"` and forces whichever
  is more likely.

Between parameters, `,` is forced; after the last one, `"}}"` closes the
`parameters` object and the outer JSON object. The method returns a plain
dict — `{"prompt", "name", "parameters"}` — which `write_results()` later
serializes.

### 6. Writing the output

Back in `main()`, results accumulate in a list as each prompt is processed
(with `rich`-powered per-prompt panels from `display.py` showing the parsed
result and timing). Once all prompts are done, `write_results()` dumps the
list to `data/output/function_calling_results.json` with `json.dump(...,
indent=2)`, and a summary table (prompts processed, successes, failures,
accuracy, timing) is printed.

## Algorithm explanation: constrained decoding

At every decoding step the model produces one logit per vocabulary token.
Normally you'd soften-max these and sample, or just take the argmax. Here,
before that happens, an `allowed_ids` set is computed from the current
generation state (which function names are still possible, which characters
can legally extend the number being built, etc.), every logit **not** in
that set is forced to `-inf` in `_constrained_decoding()`, and only then is the
argmax taken. Because the impossible tokens are removed from the
distribution entirely rather than merely discouraged, the output is
guaranteed — not just likely — to be valid JSON that matches the function
schema, no matter how weak the underlying model's raw predictions are.

The literal, unavoidable parts of the JSON scaffolding (`{"prompt":"`,
`","name":"`, `","parameters":{`, the commas between parameters, the closing
`"}}"`) are never left to the model at all — they're injected directly with
inline `self._input_ids.extend(self._encode(text))` calls. The model is only
asked to make the decisions that actually require intelligence: which
function fits the prompt, and what values its arguments should take.

## Design decisions

- **Vocabulary pre-decoding.** Decoding every token ID once during
  `initialize()` and caching it in `_id_to_decoded` trades a small amount of
  startup time for much faster generation, since no repeated tokenizer calls
  are needed per generated token.
- **Precomputed numeric/terminator token sets.** Same idea — scanning the
  vocabulary for "number-safe" tokens once instead of per digit keeps number
  generation fast.
- **Trie-based name matching.** Rather than asking the model to freely
  generate a string and then checking whether it happens to match a real
  function name, the set of valid function names is encoded as a live trie
  of allowed next-tokens, so an invalid function name is never even
  representable.
- **Forcing the JSON scaffolding.** Every character that doesn't require a
  decision (quotes, braces, colons, commas, the `parameters` key, etc.) is
  injected directly instead of generated, which removes an entire class of
  potential malformed-JSON failures and also saves generation steps.
- **pydantic everywhere.** Both the input file schemas (`FunctionDefinition`,
  `PromptEntry`, `TypeSchema`) and the generator's own parameters
  (`FunctionCallGenerator`) are pydantic models, so malformed input is
  rejected with a descriptive error before any model inference is attempted.

## Performance analysis

- **Validity:** 100% of outputs are valid, parseable JSON that matches the
  requested schema, by construction — invalid tokens are masked out, not
  merely discouraged, so there is no failure mode where the model "goes off
  script."
- **Accuracy:** function selection is effectively deterministic given the
  prompt (argmax over a masked trie), and argument extraction quality
  depends on how well the base model's raw logits track the numbers/strings
  actually present in the prompt.
- **Speed:** most of the wall-clock time per prompt goes into repeated
  forward passes of the model (one per generated token, per
  `get_logits_from_input_ids` call), not into the constraint logic itself,
  which is just array masking and an argmax.

## Challenges faced

- The tokenizer's byte-level BPE representation means tokens aren't clean
  characters — a "digit" or a `"` can be buried inside a larger token
  alongside other text (e.g. a token that ends a string might contain
  trailing characters after the closing quote). `_generate_string()` handles
  this by splitting on the first `"` it sees inside a token and only forcing
  back the text that belongs before it.
- Precomputing which tokens are safe for numbers up front (instead of
  re-parsing candidate tokens on every digit) was necessary to keep number
  generation fast, since the vocabulary is scanned once in `initialize()`
  rather than on every generation step.
- Making sure invalid states are unreachable rather than merely unlikely
  required masking logits to `-inf` rather than, e.g., down-weighting them —
  a discouraged-but-not-impossible token would eventually get picked given
  the wrong random seed or prompt.

## Testing strategy

The project was validated by running it end-to-end against the provided
sample data (`data/input/functions_definition.json` and
`data/input/function_calling_tests.json`, covering `fn_add_numbers`,
`fn_greet`, `fn_reverse_string`, `fn_get_square_root`, and
`fn_substitute_string_with_regex`) and manually inspecting
`data/output/function_calling_results.json` for:

- Valid, parseable JSON on every run.
- Correct function name chosen for each prompt.
- Correct argument types (numbers as `number`/`integer`, strings quoted
  correctly) matching each function's declared schema.
- Graceful handling of edge cases: missing input files, malformed JSON,
  empty prompt/function arrays, and prompts that don't map cleanly to any
  function.

## Multi-model support

Although the subject only requires `Qwen/Qwen3-0.6B`, the generation pipeline
never talks to `transformers` directly — it only calls the four methods
exposed by the `llm_sdk` interface (`encode`, `decode`,
`get_logits_from_input_ids`, `get_path_to_vocab_file`). Because
`FunctionCallGenerator` is written against that interface and not against a
specific class, swapping the backing model is just a matter of providing
another object with the same shape.

`src/__main__.py` demonstrates this with a small `models` registry mapping a
name to a zero-argument constructor:

```python
models: dict[str, Callable[[], Any]] = {
    "Qwen": Small_LLM_Model,
    "SmolLM2": SmolLM2Backend,
}
```

`SmolLM2Backend` (in `smollm2_backend.py`) wraps `HuggingFaceTB/SmolLM2` behind
the exact same four-method contract as `Small_LLM_Model`, so it can be
instantiated and passed straight into `FunctionCallGenerator` with zero
changes to `generator.py`, `_generate_name()`, `_generate_number()`, or any
other constrained-decoding logic. Selecting a backend is currently a
one-line change (`models["SmolLM2"]` instead of `models["Qwen"]`), but the
registry is already shaped so it could just as easily be driven by a
`--model` CLI flag.

The only real requirement for adding a new backend is that its tokenizer's
vocabulary is exposed the same way `get_path_to_vocab_file()` exposes Qwen's
— everything downstream (numeric-token precomputation, trie-based name
matching, JSON forcing) is model-agnostic and works unchanged.

## Example usage

```bash
uv sync
uv run python -m src
```

Input (`data/input/function_calling_tests.json`, excerpt):

```json
[
  { "prompt": "What is the sum of 2 and 3?" },
  { "prompt": "Greet shrek" },
  { "prompt": "Reverse the string 'hello'" }
]
```

Output (`data/output/function_calling_results.json`):

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": { "a": 2.0, "b": 3.0 }
  },
  {
    "prompt": "Greet shrek",
    "name": "fn_greet",
    "parameters": { "name": "shrek" }
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": { "s": "hello" }
  }
]
```

## Bonus Implementations

The project includes several bonus features and improvements beyond the core requirements:

* **Support for multiple LLM models**
  The project supports multiple language model backends, including **Qwen** and **SmolLM2**, through a common model selection mechanism.

* **Recoded tokenizer integration**
  The main generation code avoids directly depending on the tokenizer's `encode` and `decode` methods. Instead, token IDs, vocabulary data, decoded token mappings, and logits are handled through the LLM SDK and internal token mappings.

* **Advanced error recovery mechanisms**
  The implementation includes error handling through `try/except` blocks, `RuntimeError` checks for stuck generation states, active-set exhaustion handling, and a safety fallback for string generation. These mechanisms provide robust failure detection and handling, although they focus more on error handling and failure prevention than on advanced retry or recovery strategies.


* **Performance optimizations (caching, batching)**
  The implementation includes caching optimizations through `_id_to_decoded`, which avoids repeatedly decoding the same tokens, and precomputes `_numeric_ids` and `_terminator_ids` to reduce repeated token filtering and lookup overhead during generation. Batching is not implemented.


* **Visualization of the generation process**
  The generation process is presented interactively in the terminal using the **Rich** library. Results are displayed prompt by prompt with structured panels showing the original prompt, selected function, generated parameters, JSON output, timing information, errors, and an overall execution summary.

* **Encoding and constrained decoding demonstration**
  The project demonstrates how text is converted into token IDs, how token IDs are used to obtain model logits, and how constrained decoding uses those logits to select only valid tokens before reconstructing the final function call.

These additions make the project more modular, extensible, and easier to debug while demonstrating a deeper understanding of tokenization, model inference, and constrained decoding.

## Resources

- [Hugging Face — Text generation strategies](https://huggingface.co/docs/transformers/generation_strategies)
- [Hugging Face — `Qwen/Qwen3-0.6B` model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [pydantic v2 documentation](https://docs.pydantic.dev/latest/)
- [`uv` documentation](https://docs.astral.sh/uv/)
- Background reading on constrained/guided decoding for structured LLM
  output (the general technique behind tools like Outlines and
  guidance-style JSON-schema-constrained generation).

**AI usage:** AI assistance was used to help think through and explain the
constrained-decoding pipeline (trie-based name matching, per-type value
generation, token masking) while writing this README, and earlier in the
project to reason about tokenizer edge cases (BPE leading-space markers,
splitting quote characters out of decoded tokens). All code in `src/` was
written, tested, and understood directly rather than generated wholesale, in
line with the project's AI usage guidelines.
