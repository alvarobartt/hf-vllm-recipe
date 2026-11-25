from hf_vllm_recipe.reference import ReleaseReference


def example_reference() -> ReleaseReference:
    native = (
        "Qwen3ForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Gemma4ForConditionalGeneration",
        "MistralForCausalLM",
        "PixtralForConditionalGeneration",
        "DeepseekV32ForCausalLM",
        "DeepseekV4ForCausalLM",
    )
    return {
        "version": "0.34.0",
        "transformers_version": "5.16.1",
        "architectures": {**dict.fromkeys(native, "vllm"), "GemmaForCausalLM": "transformers"},
        "tool_parsers": [
            "hermes",
            "hf",
            "mistral",
            "gemma4",
            "qwen3_xml",
            "qwen3_coder",
            "deepseek_v32",
            "deepseek_v4",
        ],
        "reasoning_parsers": [
            "qwen3",
            "deepseek_r1",
            "hf",
            "mistral",
            "gemma4",
            "deepseek_v3",
            "deepseek_v4",
        ],
        "config_types": [
            "qwen3",
            "qwen3_5",
            "gemma4",
            "gemma",
            "mistral",
            "pixtral",
            "deepseek_v32",
            "deepseek_v4",
        ],
        "tokenizer_classes": [
            "Qwen2TokenizerFast",
            "Qwen2Tokenizer",
            "GemmaTokenizer",
            "LlamaTokenizerFast",
            "PreTrainedTokenizerFast",
        ],
        "processor_classes": [],
        "sources": {},
    }
