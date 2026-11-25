---
name: hf-vllm-recipe
description: Generate a baseline vLLM serve command for a Hugging Face model or local checkpoint when the user needs context, tokenizer, loader, tool-calling or reasoning settings.
---

Generate the serving command with the CLI rather than guessing model-specific flags:

```bash
uvx hf-vllm-recipe MODEL
```

Before the first PyPI release, run it from the repository:

```bash
uvx --from git+https://github.com/alvarobartt/hf-vllm-recipe hf-vllm-recipe MODEL
```

- The default vLLM reference is `0.34.0`, with `--version` selecting another release.
- Use the `hf-mem` extra (`uvx 'hf-vllm-recipe[hf-mem]' MODEL --gpu-memory SIZE`) and `--gpu-memory` only when the user requests a memory estimate.
- Preserve the returned model revision, context, parser and native-format flags without adding performance tuning.
- Return the complete `vllm serve` command and report any inspection errors instead of inventing missing settings.
- Generating a command does not validate GPU startup or request starting a server.
