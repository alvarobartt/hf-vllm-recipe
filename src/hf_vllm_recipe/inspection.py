import json
import math
import os
import re
from pathlib import Path
from typing import TypeGuard
from urllib.parse import quote

import httpx2

from .reference import MAX_METADATA_BYTES, InspectionError, JsonObject, ReleaseReference, fetch_text

CONTEXT_KEYS = (
    "max_position_embeddings",
    "n_positions",
    "max_seq_len",
    "seq_length",
    "model_max_length",
    "max_target_positions",
    "max_sequence_length",
    "max_seq_length",
    "seq_len",
)


def hub_token() -> str | None:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    hf_home = Path(
        os.getenv(
            "HF_HOME",
            Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "huggingface",
        )
    )
    token_path = Path(os.getenv("HF_TOKEN_PATH", hf_home / "token"))
    try:
        return token_path.read_text(encoding="utf-8").strip() or None
    except FileNotFoundError:
        return None


class ModelSource:
    def __init__(self, model: str, revision: str | None = None, offline: bool = False) -> None:
        path = Path(model).expanduser()
        self.local = path.is_dir()
        self.model = str(path.resolve()) if self.local else model
        self.revision = revision
        self.token = None if self.local else hub_token()
        if self.local:
            if revision:
                raise InspectionError("--revision only applies to Hub models")
            self.files = {
                item.relative_to(path).as_posix() for item in path.rglob("*") if item.is_file()
            }
        else:
            if offline:
                raise InspectionError("--offline requires a local model directory")
            if not re.fullmatch(r"[\w.-]+/[\w.-]+", model) or any(
                part in (".", "..") for part in model.split("/")
            ):
                raise InspectionError(f"Not a local directory or Hub ID (org/model): {model}")
            info = self._remote_json(
                f"https://huggingface.co/api/models/{quote(model, safe='/')}/revision/{quote(revision or 'main', safe='')}"
            )
            self.files = {item["rfilename"] for item in info.get("siblings", [])}
            # Inspect and serve the same immutable revision, even if main moves.
            commit = info.get("sha")
            self.revision = commit if isinstance(commit, str) and commit else revision or "main"

    def _remote_json(self, url: str) -> JsonObject:
        try:
            result = json.loads(fetch_text(url, self.token))
            if not isinstance(result, dict):
                raise ValueError("Expected an object")
            return result
        except httpx2.HTTPStatusError as exc:
            hint = (
                "; set HF_TOKEN or run hf auth login for private/gated models"
                if exc.response.status_code in (401, 403)
                else ""
            )
            raise InspectionError(
                f"Hub request failed: HTTP {exc.response.status_code} at {url}{hint}"
            ) from exc
        except ValueError as exc:
            raise InspectionError(f"Invalid JSON at {url}") from exc

    def read(self, filename: str) -> str | None:
        if filename not in self.files:
            return None
        if self.local:
            path = Path(self.model) / filename
            if path.stat().st_size > MAX_METADATA_BYTES:
                raise InspectionError(f"Metadata file is too large: {path}")
            return path.read_text(encoding="utf-8")
        if self.revision is None:
            raise InspectionError("Hub model has no resolved revision")
        url = f"https://huggingface.co/{quote(self.model, safe='/')}/resolve/{quote(self.revision, safe='')}/{quote(filename, safe='/')}"
        try:
            return fetch_text(url, self.token)
        except httpx2.HTTPStatusError as exc:
            raise InspectionError(
                f"Cannot read {filename}: HTTP {exc.response.status_code}; gated models may require HF_TOKEN"
            ) from exc

    def json(self, filename: str, required: bool = False) -> JsonObject:
        text = self.read(filename)
        if text is None:
            if required:
                raise InspectionError(f"{filename} is missing from {self.model}")
            return {}
        try:
            result = json.loads(text)
            if not isinstance(result, dict):
                raise ValueError("Expected an object")
            return result
        except ValueError as exc:
            raise InspectionError(f"Invalid {filename}: {exc}") from exc


def text_config(config: JsonObject) -> JsonObject:
    for key in (
        "text_config",
        "llm_config",
        "language_config",
        "language_model_config",
    ):
        if isinstance(config.get(key), dict):
            return text_config(config[key])
    if isinstance(config.get("thinker_config"), dict):
        return text_config(config["thinker_config"])
    return config


def positive_int(value: object) -> TypeGuard[int]:
    return type(value) is int and 0 < value < 10**12


def context_length(config: JsonObject, tokenizer: JsonObject, warnings: list[str]) -> int:
    text = text_config(config)
    values = [value for key in CONTEXT_KEYS if positive_int(value := text.get(key))]
    # Cohere's model_max_length explicitly overrides other context fields in vLLM.
    configured = text.get("model_max_length")
    length = configured if positive_int(configured) else min(values, default=None)
    if length is None:
        length = tokenizer.get("model_max_length")
        if not positive_int(length):
            raise InspectionError(
                "No context length in config.json or tokenizer_config.json; supply --max-model-len"
            )
        warnings.append(
            "Context length comes from tokenizer_config.json; no model context field was found"
        )
    rope = text.get("rope_parameters") or text.get("rope_scaling") or {}
    if not isinstance(rope, dict):
        raise InspectionError("RoPE parameters must be an object in config.json")
    parameters = (
        [rope]
        if "rope_type" in rope or "type" in rope
        else [item for item in rope.values() if isinstance(item, dict)]
    )
    factor = 1.0
    for item in parameters:
        kind = item.get("rope_type", item.get("type", "default"))
        if kind == "longrope" and positive_int(text.get("original_max_position_embeddings")):
            return text["original_max_position_embeddings"]
        if kind not in (
            "su",
            "longrope",
            "llama3",
            "yarn",
            "deepseek_yarn",
            "deepseek_llama_scaling",
        ):
            value = item.get("factor", 1.0)
            if value is None:
                value = 1.0
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise InspectionError("Invalid RoPE scaling factor in config.json")
            factor = value
    if "gemma3" not in str(text.get("model_type", config.get("model_type", ""))):
        length = int(length * factor)
    return length


def architecture(
    config: JsonObject, reference: ReleaseReference, requested: str
) -> tuple[str, str]:
    architectures = config.get("architectures") or text_config(config).get("architectures") or []
    if not isinstance(architectures, list) or not all(
        isinstance(item, str) for item in architectures
    ):
        raise InspectionError("config.json architectures must be a list of strings")
    if requested == "transformers":
        return next(iter(architectures), "unknown"), "transformers"
    # vLLM checks all explicitly registered architectures before generic fallback.
    for name in architectures:
        implementation = reference["architectures"].get(name)
        if implementation and (requested == "auto" or implementation == requested):
            return name, implementation
    names = ", ".join(architectures) or "(no architectures declared)"
    raise InspectionError(
        f"Architecture {names} has no matching {requested} implementation registered in vLLM {reference['version']}. "
        "Use --model-impl transformers for an unverified Transformers-backend command, if the model supports it"
    )


def custom_code(
    configs: dict[str, JsonObject],
    files: set[str],
    reference: ReleaseReference,
    implementation: str,
) -> list[str]:
    reasons = []
    known_configs = set(reference["config_types"])
    known_tokenizers = set(reference["tokenizer_classes"])
    known_processors = set(reference["processor_classes"])
    tokenizer_class = configs.get("tokenizer_config.json", {}).get("tokenizer_class")
    processor_class = next(
        (item.get("processor_class") for item in configs.values() if item.get("processor_class")),
        None,
    )

    def walk(config: JsonObject, location: str) -> None:
        mapping = config.get("auto_map") or {}
        if not isinstance(mapping, dict):
            raise InspectionError(f"Invalid auto_map in {location}")
        for auto_class, target in mapping.items():
            needed = (
                (auto_class == "AutoConfig" and config.get("model_type") not in known_configs)
                or (
                    auto_class == "AutoTokenizer"
                    and (tokenizer_class or config.get("tokenizer_class")) not in known_tokenizers
                )
                or (
                    auto_class == "AutoProcessor"
                    and (processor_class or config.get("processor_class")) not in known_processors
                )
                or (
                    auto_class.startswith("AutoModel")
                    and implementation == "transformers"
                    and config.get("model_type") not in known_configs
                )
            )
            if not needed or not target:
                continue
            targets = target if isinstance(target, (list, tuple)) else [target]
            for name in targets:
                if not isinstance(name, str):
                    continue
                external = "--" in name
                module = name.rsplit("--", 1)[-1].rsplit(".", 1)[0].replace(".", "/") + ".py"
                evidence = (
                    "external repository"
                    if external
                    else "file present"
                    if module in files
                    else "referenced file not listed"
                )
                reasons.append(f"{location}: {auto_class} -> {name} ({evidence})")
        for key, value in config.items():
            if key.endswith("_config") and isinstance(value, dict):
                walk(value, f"{location}.{key}")

    for filename, config in configs.items():
        walk(config, filename)
    return reasons


def chat_templates(source: ModelSource, tokenizer: JsonObject) -> tuple[str, str]:
    templates: dict[str, str] = {}
    legacy = source.json("chat_template.json")
    embedded = legacy.get("chat_template", tokenizer.get("chat_template"))
    if isinstance(embedded, str):
        templates["default"] = embedded
    elif isinstance(embedded, dict):
        templates.update({key: value for key, value in embedded.items() if isinstance(value, str)})
    elif isinstance(embedded, list):
        for item in embedded:
            if isinstance(item, dict) and isinstance(item.get("template"), str):
                templates[item.get("name", "default")] = item["template"]
    standalone = source.read("chat_template.jinja")
    if standalone is not None:
        templates["default"] = standalone
    for filename in sorted(source.files):
        if filename.startswith("chat_templates/") and filename.endswith(".jinja"):
            named_template = source.read(filename)
            if named_template is not None:
                templates[Path(filename).stem] = named_template
    if not templates:
        # Native DeepSeek encoders supply format markers instead of Jinja.
        # Read them as text; never import or execute checkpoint Python.
        for filename in (
            "encoding/encoding_dsv41.py",
            "encoding/encoding_dsv4.py",
            "encoding/encoding_dsv32.py",
        ):
            encoder = source.read(filename)
            if (
                encoder
                and "｜DSML｜" in encoder
                and ("function_calls" in encoder or "tool_calls" in encoder)
            ):
                return encoder, encoder
        return "", ""
    default = templates.get("default")
    if default is None and len(templates) == 1:
        default = next(iter(templates.values()))
    # Transformers selects tool_use for requests containing tools.
    return templates.get("tool_use", default or ""), default or ""


def infer_parsers(
    tool_template: str, reasoning_template: str, family: str, tokenizer: JsonObject
) -> tuple[str | None, str | None]:
    tool = reasoning = None
    response = tokenizer.get("response_template")
    if isinstance(response, dict):
        tool = "hf" if response.get("tool_calls") else None
        reasoning = "hf" if response.get("thinking") else None
        if tool or reasoning:
            return tool, reasoning
    # Check specific envelopes before the generic Hermes JSON envelope.
    deepseek_v4 = re.search(r"deepseek[-_]?v4", family)
    deepseek_v41 = re.search(r"deepseek[-_]?v4(?:1|\.1)", family)
    if "｜DSML｜" in tool_template and (
        "function_calls" in tool_template or "tool_calls" in tool_template
    ):
        tool = "deepseek_v41" if deepseek_v41 else "deepseek_v4" if deepseek_v4 else "deepseek_v32"
    elif "<|tool_call_begin|>" in tool_template or "<|tool_calls_section_begin|>" in tool_template:
        tool = "kimi_k2"
    elif "<｜tool▁calls▁begin｜>" in tool_template:
        tool = "deepseek_v31" if "v3.1" in family or "v31" in family else "deepseek_v3"
    elif "<minimax:tool_call>" in tool_template:
        tool = "minimax_m2"
    elif "[TOOL_CALLS]" in tool_template:
        tool = "mistral"
    elif "<|action_start|>" in tool_template and "<|plugin|>" in tool_template:
        tool = "internlm"
    elif "<start_function_call>" in tool_template:
        tool = "functiongemma"
    elif "<tool_calls>" in tool_template and "</tool_calls>" in tool_template:
        tool = "jamba"
    elif "<|python_tag|>" in tool_template:
        tool = "llama4_json" if "llama4" in family else "llama3_json"
    elif "<tool_call>" in tool_template:
        if "<arg_key>" in tool_template and "glm" in family:
            tool = "glm47"
        elif "<function=" in tool_template:
            tool = "granite_toolcall_parser" if "granite" in family else "qwen3_xml"
        elif "granite" in family:
            tool = "granite4"
        else:
            tool = "hermes"
    elif "<|tool_call>" in tool_template and "gemma4" in family.replace("-", ""):
        tool = "gemma4"
    elif "<|tool_call|>" in tool_template and "granite" in family:
        tool = "granite"
    elif "<|start|>" in tool_template and "<|channel|>" in tool_template and "to=" in tool_template:
        tool = "openai"
    content = reasoning_template or tool_template
    if "<|start|>" in content and "<|channel|>" in content and "analysis" in content:
        reasoning = "openai_gptoss"
    elif "<|channel>" in content and ("thought" in content or "<|think|>" in content):
        reasoning = "gemma4" if "gemma4" in family.replace("-", "") else None
    elif "[THINK]" in content or "[/THINK]" in content:
        reasoning = "mistral"
    elif "<think>" in content or "</think>" in content:
        if deepseek_v41:
            reasoning = "deepseek_v41"
        elif deepseek_v4:
            reasoning = "deepseek_v4"
        elif "deepseek-r1" in family or "qwq" in family:
            reasoning = "deepseek_r1"
        elif "minimax" in family:
            reasoning = "minimax_m2_append_think"
        elif "qwen3" in family:
            reasoning = "qwen3"
        elif "glm" in family:
            reasoning = "glm47"
        elif "deepseek" in family and ("thinking" in content):
            reasoning = "deepseek_v3"
        elif "kimi" in family:
            reasoning = "kimi_k2"
        elif "granite" in family:
            reasoning = (
                "granite_thinking_parser"
                if tool == "granite_toolcall_parser" or "enable_thinking" in content
                else "granite"
            )
        else:
            reasoning = "deepseek_r1"
    elif "<|thought|>" in content and "granite" in family:
        reasoning = "granite"
    return tool, reasoning
