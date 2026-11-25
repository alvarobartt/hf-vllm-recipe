# `hf-vllm-recipe`

`hf-vllm-recipe` is a Hugging Face CLI extension that generates a baseline `vllm serve` command for a Hub model or local checkpoint.

- Read context length and check support against vLLM `0.34.0` or `--version`
- Detect custom code, chat parsers and native checkpoint formats
- Estimate memory with [hf-mem](https://github.com/alvarobartt/hf-mem) using `--gpu-memory` and the optional `hf-mem` extra
- Python 3.12+ and `httpx2`

## Installation

```bash
hf extensions install alvarobartt/hf-vllm-recipe
```

## Usage

Run with `uvx`, or use `hf vllm-recipe MODEL` after installing the extension.

```console
$ uvx hf-vllm-recipe google/gemma-4-E2B-it
vllm serve google/gemma-4-E2B-it \
  --max-model-len 131072 \
  --revision 4adf15748de9e1a0097b6274cee9c15836c41c37 \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --reasoning-parser gemma4
```

## Agents

Coding agents can follow the [`hf-vllm-recipe` skill](skills/hf-vllm-recipe/SKILL.md) to generate serving commands.
