from pathlib import PurePosixPath

from .inspection import ModelSource, text_config
from .reference import InspectionError, JsonObject


def model_config(source: ModelSource) -> tuple[JsonObject, dict[str, str]]:
    config = source.json("config.json")
    consolidated = any(
        PurePosixPath(name).match("consolidated*.safetensors") for name in source.files
    )
    if not (consolidated and "params.json" in source.files):
        if not config:
            raise InspectionError(f"config.json is missing from {source.model}")
        return config, {}

    params = source.json("params.json", required=True)
    # Inspect only fields used for architecture/context selection. vLLM performs
    # the full native-to-HF conversion at startup; no converted file is written.
    text = {**text_config(config), **params}
    for native, hf in (
        ("dim", "hidden_size"),
        ("n_heads", "num_attention_heads"),
        ("n_kv_heads", "num_key_value_heads"),
        ("n_layers", "num_hidden_layers"),
        ("hidden_dim", "intermediate_size"),
    ):
        if native in params:
            text[hf] = params[native]
    text.setdefault("max_position_embeddings", 128000)
    text.setdefault("max_seq_len", text["max_position_embeddings"])
    moe = params.get("moe") or {}
    if not isinstance(moe, dict):
        raise InspectionError("Invalid Mistral moe configuration in params.json")
    if params.get("model_type") == "mamba":
        native_arch, native_type = "Mamba2ForCausalLM", "mamba2"
    elif params.get("dsa"):
        native_arch, native_type = "MistralLarge4ForCausalLM", "mistral_large_4"
    elif moe.get("num_shared_experts"):
        native_arch, native_type = "MistralLarge3ForCausalLM", "deepseek_v3"
    elif moe:
        native_arch, native_type = "MixtralForCausalLM", "mixtral"
    else:
        native_arch, native_type = "MistralForCausalLM", "mistral"
    text["architectures"] = [native_arch]
    text["model_type"] = native_type
    multimodal = params.get("multimodal") or {}
    if not isinstance(multimodal, dict):
        raise InspectionError("Invalid Mistral multimodal configuration in params.json")
    if params.get("vision_encoder") or multimodal.get("vision_encoder_args"):
        config = {
            "architectures": ["PixtralForConditionalGeneration"],
            "model_type": "pixtral",
            "text_config": text,
        }
    elif multimodal.get("whisper_model_args"):
        # Audio models need their declared Voxtral architecture; do not guess.
        if not config.get("architectures"):
            raise InspectionError(
                "Native Mistral audio inspection requires config.json architectures"
            )
        config = {**config, "text_config": text}
    else:
        config = text

    native_tokenizer = "tekken.json" in source.files or any(
        PurePosixPath(name).match("tokenizer.model*") for name in source.files
    )
    if not native_tokenizer and "tokenizer.json" not in source.files:
        raise InspectionError("Mistral weights found without a native or HF tokenizer")
    return config, {
        "tokenizer-mode": "mistral" if native_tokenizer else "hf",
        "config-format": "mistral",
        "load-format": "mistral",
    }


def tokenizer_mode(config: JsonObject) -> str | None:
    modes = {
        "DeepseekV32ForCausalLM": "deepseek_v32",
        "DeepseekV4ForCausalLM": "deepseek_v4",
        "DeepseekV4ForConditionalGeneration": "deepseek_v4",
        "DeepseekV41ForCausalLM": "deepseek_v41",
    }
    for name in config.get("architectures", []):
        if name in modes:
            return modes[name]
    return None


def native_tool_parser(
    source: ModelSource, tokenizer: JsonObject, formats: dict[str, str]
) -> str | None:
    if formats.get("tokenizer-mode") != "mistral":
        return None
    # Mistral Common renders tools natively. Older HF Jinja templates can lack
    # tool instructions even though the native tokenizer has tool delimiters.
    for metadata in (tokenizer, source.json("special_tokens_map.json")):
        specials = metadata.get("additional_special_tokens") or []
        for value in specials:
            if value == "[TOOL_CALLS]" or (
                isinstance(value, dict) and value.get("content") == "[TOOL_CALLS]"
            ):
                return "mistral"
        added = metadata.get("added_tokens_decoder") or {}
        if isinstance(added, dict) and any(
            isinstance(value, dict) and value.get("content") == "[TOOL_CALLS]"
            for value in added.values()
        ):
            return "mistral"
    return None
