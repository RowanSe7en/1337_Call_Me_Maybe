Move uv cache into goinfre, store downloaded packages and temporary extraction files here:

```bash id="sr6t0n"
export UV_CACHE_DIR="/home/$USER/goinfre/uv-cache"
```

Move HuggingFace models/cache into goinfre:

```bash id="1lj5p7"
export HF_HOME="/home/$USER/goinfre/huggingface"
```

Move the Python virtual environment into goinfre:

```bash id="wjjqxp"
export UV_PROJECT_ENVIRONMENT="/home/$USER/goinfre/call/.venv"
```

Create the uv cache, HuggingFace cache, directories:

```bash id="2d0b8e"
mkdir -p $UV_CACHE_DIR
mkdir -p $HF_HOME
```

Create the folder that will contain the virtualenv:

```bash id="4xbm1g"
mkdir -p /home/$USER/goinfre/call
```

Make uv cache, HuggingFace cache, and the virtualenv location permanent in every new shell:

```bash id="y55zhm"
echo 'export UV_CACHE_DIR="/home/$USER/goinfre/uv-cache"' >> ~/.zshrc
echo 'export HF_HOME="/home/$USER/goinfre/huggingface"' >> ~/.zshrc
echo 'export UV_PROJECT_ENVIRONMENT="/home/$USER/goinfre/call/.venv"' >> ~/.zshrc
```

Reload `.zshrc` immediately without reopening the terminal:

```bash id="rr6uy8"
source ~/.zshrc
```

installation with the new paths:

```bash id="v22a1h"
make install
```








What is PyTorch (torch)?



PyTorch is a Python library primarily used for:



Machine Learning

Deep Learning

Running neural networks

Training models

Inference (using already-trained models)



What is PyTorch used for in this file?



Not training.



This file only uses PyTorch to:



Load an already-trained LLM.

Move it to CPU/GPU.

Create tensors from token IDs.

Run inference.

Disable gradients to save memory.


1) What is device?



In PyTorch, a device means:



the hardware where tensors and model computations will live and run



Examples of devices:



"cpu" → run on the processor

"cuda" → run on an NVIDIA GPU

"mps" → run on Apple Silicon GPU through Metal



So if you have:



x = torch.tensor([1, 2, 3], device="cuda")



that tensor is stored on the GPU, and operations on it happen there.



2) What is dtype?



dtype means:



the numeric data type used to store numbers inside tensors/model weights



Examples:



torch.float32 → 32-bit floating point

torch.float16 → 16-bit floating point

torch.long → 64-bit integer



So dtype answers:



“What format should the numbers use in memory?”

Example:



torch.tensor([1.5, 2.7], dtype=torch.float32)

float16 good for GPUs?



Because GPUs are built to do huge numbers of parallel math operations efficiently, and modern GPUs often handle float16 very well.

Why not use float16 on CPU too?



Because CPUs generally do not benefit from float16 the same way GPUs do.



Possible issues:



A) Worse support / slower operations



Many CPU operations are best optimized for float32, not float16.



So using float16 on CPU may be slower or less supported.

Why is it torch.backends.mps.is_available() and not torch.mps.is_available()?



Because MPS is exposed by PyTorch as a backend module, not as a top-level main namespace like torch.cuda.



So structurally, PyTorch decided:

CUDA is NVIDIA’s platform for running code on NVIDIA GPUs.

CUDA gets a top-level API:



torch.cuda



MPS = Metal Performance Shaders



It’s Apple’s GPU compute system used by PyTorch on Apple Silicon machines.MPS availability is exposed under:



torch.backends.mps



This is mostly an API design choice by PyTorch.


















































































“Load the tokenizer associated with this Hugging Face model.”

self._tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(

    model_name, trust_remote_code=trust_remote_code

)

A tokenizer is the component that converts text into token IDs that the model understands.

model_name = "Qwen/Qwen3-0.6B"



then Hugging Face looks up that model repo and downloads/loads the tokenizer files for it.

So after this line:



self._tokenizer



becomes an object that can do things like:



self._tokenizer.encode("hello")

self._tokenizer.decode([123, 456])





What is a pad token?



A pad token is a special token used to make sequences the same length.



Their tokenized lengths may differ:



[12, 55]              # length 2

[12, 55, 91, 300]     # length 4



To put them in one batch, they often need equal length, so you pad the shorter one:



[12, 55, PAD, PAD]

[12, 55, 91, 300]



The tokenizer needs to know which token ID is the pad token.



Some causal language models do not define a separate pad token.



So:



self._tokenizer.pad_token_id



may be None.



That can break helper functions that expect a pad token to exist.



So this code says:



“If the tokenizer has no pad token, use the EOS token as the padding token.”



3) Load the model

“Download/load the pretrained weights and build the corresponding PyTorch model object.”

self._model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(

    model_name,

    torch_dtype=self._dtype,

    device_map="auto" if self._device == "cuda" else None,

    trust_remote_code=trust_remote_code,

)



This is the line that loads the actual neural network.



What is AutoModelForCausalLM?



It’s a Hugging Face factory class that says:



“Load a model suitable for causal language modeling.”



Causal LM means a model that predicts the next token from previous tokens.



Examples of causal LMs:



GPT-style models

Qwen

LLaMA

Mistral



So this class loads the proper model architecture for the chosen repo.

device_map tells Hugging Face how to place the model on available devices.

If trust_remote_code=True, Hugging Face is allowed to use that custom code.



4) Move the model to the chosen device

self._model.to(self._device)



Put model in evaluation mode

self._model.eval()



This tells the model:



“We are using it for inference, not training.”



Disable gradients for every parameter

for p in self._model.parameters():

    p.requires_grad = False



In PyTorch, tensors/parameters can track gradients for training.



If:



p.requires_grad = True



But here you are not training.



You are only doing inference.



So the code sets:



p.requires_grad = False




















































































1) encode

def encode(self, text: str) -> torch.Tensor:

    """Tokenise *text* and return a 2-D ``input_ids`` tensor on the target device."""

    ids = self._tokenizer.encode(text, add_special_tokens=False)

    return torch.tensor([ids], device=self._device, dtype=torch.long)



encode() converts normal text into a PyTorch tensor of token IDs that the model can use as input.



So:



"hello world"



becomes something like:



tensor([[9707, 1879]])

Why [ids] and not just ids?



Because models expect input in shape:



[batch_size, sequence_length]



If you passed only:



torch.tensor(ids)



you’d get a 1-D tensor like:



tensor([9707, 1879])



But the model expects:



tensor([[9707, 1879]])



for a batch of 1 prompt.

2) decode
def decode(self, ids: torch.Tensor | list[int]) -> str:
    """Inverse of :py:meth:`encode`. Removes special tokens."""
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    return self._tokenizer.decode(ids, skip_special_tokens=True)

decode() does the reverse of encode().

It converts token IDs back into text.

Example:

[9707, 1879]

becomes something like:

"hello world"

If it’s a tensor, convert it to a normal Python list
if isinstance(ids, torch.Tensor):
    ids = ids.tolist()

Example:

tensor([9707, 1879])

becomes:

[9707, 1879]

Why?

Because the tokenizer’s decode() expects a list-like sequence of token IDs, and converting to a normal Python list is simple and safe.

Convert token IDs back into text



3) get_logits_from_input_ids
def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
    """
    Given a list of input token ids, return the raw logits (no softmax) for the next token.
    """
    input_tensor = torch.tensor([input_ids], device=self._device, dtype=torch.long)
    with torch.no_grad():
        out = self._model(input_ids=input_tensor)
    # Get logits for the last token in the sequence for the batch (batch size 1)
    logits = out.logits[0, -1].tolist()
    return [float(x) for x in logits]

This is the most important one.

Purpose of get_logits_from_input_ids

Given some already-tokenized text, it asks the model:

“Based on these input tokens, what score do you assign to every possible next token?”

It returns the raw next-token scores.

These scores are called logits.

What is a logit?

A logit is the raw score the model gives before probabilities are computed.

Suppose the vocabulary is:

["cat", "dog", "house", "blue", ...]

After reading your prompt, the model might produce scores like:

[2.3, -1.1, 0.7, 5.2, ...]

Each number corresponds to one token in the vocabulary.

Higher score = model thinks that token is more likely as the next token.

These are not probabilities yet.
Softmax would convert them into probabilities, but this function does not do that.

Run the model without gradients
with torch.no_grad():
    out = self._model(input_ids=input_tensor)

This sends the tokens through the language model.

means roughly:

tell PyTorch: stop tracking gradients
run the model
restore the previous gradient-tracking state when done

What is this line doing?
out = self._model(input_ids=input_tensor)

This is basically calling the model like a function.

It means:

“Run a forward pass of the language model using these input token IDs.”

So if:

input_tensor = tensor([[9707, 1879]])

the model processes those tokens and returns an output object out.

That output contains things like:

out.logits

which are the next-token scores.
Its shape is:

[batch_size, sequence_length, vocab_size]
Suppose your input is:

input_ids = [10, 20, 30]

Then the code makes:

input_tensor = torch.tensor([[10, 20, 30]])

Shape of input_tensor:

[1, 3]

because:

batch size = 1
sequence length = 3

When the model runs, it returns logits for each token position.

So out.logits shape becomes:

[1, 3, vocab_size]

Extract logits for the last input position
logits = out.logits[0, -1].tolist()
example:
out.logits =
tensor([
    [
        [0.1, 0.2, 0.3, 0.4, 0.5],   # position 0
        [1.0, 1.1, 1.2, 1.3, 1.4],   # position 1
        [2.0, 2.1, 2.2, 2.3, 2.4]    # position 2
    ]
])
out.logits[0, -1]

means:

take batch 0
then take the last position in that sequence

out.logits[0, -1]

becomes:

tensor([2.0, 2.1, 2.2, 2.3, 2.4])




Briefly:

These **3 functions do the same kind of job**:

> **find and return the local path of a tokenizer-related file from the Hugging Face model repo**

They do **not** read the file contents.
They just return **where the file is stored on disk** after Hugging Face downloads/fetches it.

---

# 1) `get_path_to_vocab_file()`

```python
def get_path_to_vocab_file(self) -> str:
    vocab_file_name = self._tokenizer.vocab_files_names.get('vocab_file', "vocab.json")
    vocab_path = hf_hub_download(
        repo_id=self._model_name,
        filename=vocab_file_name
    )
    return vocab_path
```

## What it does

* looks up the tokenizer’s **vocab file name**
* downloads/finds that file from the model repo on Hugging Face
* returns the **local file path**

## Usually this file is

* `vocab.json`

## Returns something like

```python
"/home/user/.cache/huggingface/.../vocab.json"
```

---

# 2) `get_path_to_merges_file()`

```python
def get_path_to_merges_file(self) -> str:
    merges_file_name = self._tokenizer.vocab_files_names.get('merges_file', "merges.txt")
    merges_path = hf_hub_download(
        repo_id=self._model_name,
        filename=merges_file_name
    )
    return merges_path
```

## What it does

* looks up the tokenizer’s **merges file name**
* downloads/finds it from the Hugging Face repo
* returns the **local file path**

## Usually this file is

* `merges.txt`

## Why it exists

Some tokenizers use **BPE merge rules**, and those rules are stored in this file.

---

# 3) `get_path_to_tokenizer_file()`

```python
def get_path_to_tokenizer_file(self) -> str:
    tokenizer_file_name = self._tokenizer.vocab_files_names.get('tokenizer_file', "tokenizer.json")
    tokenizer_path = hf_hub_download(
        repo_id=self._model_name,
        filename=tokenizer_file_name
    )
    return tokenizer_path
```

## What it does

* looks up the tokenizer’s **main tokenizer file name**
* downloads/finds it from the Hugging Face repo
* returns the **local file path**

## Usually this file is

* `tokenizer.json`

---

# In one line each

* **`get_path_to_vocab_file()`** → returns path to the tokenizer’s vocabulary file
* **`get_path_to_merges_file()`** → returns path to the tokenizer’s merges rules file
* **`get_path_to_tokenizer_file()`** → returns path to the tokenizer’s main tokenizer JSON file

---

# Very short summary

All 3 are **helper functions to fetch tokenizer files from Hugging Face and give you their local path**.
