import ast
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TypedDict

import httpx2

from . import __version__

DEFAULT_VERSION = "0.34.0"
MAX_METADATA_BYTES = 8 * 1024 * 1024

type JsonObject = dict[str, Any]


class ReleaseReference(TypedDict):
    version: str
    transformers_version: str
    architectures: dict[str, str]
    tool_parsers: list[str]
    reasoning_parsers: list[str]
    config_types: list[str]
    tokenizer_classes: list[str]
    processor_classes: list[str]
    sources: dict[str, str]


class InspectionError(Exception):
    """A model or release could not be inspected reliably."""


def fetch_text(url: str, token: str | None = None) -> str:
    headers = {"User-Agent": f"hf-vllm-recipe/{__version__}"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        with httpx2.Client(follow_redirects=True, timeout=30) as client:
            with client.stream("GET", url, headers=headers) as response:
                response.raise_for_status()
                data = bytearray()
                for chunk in response.iter_bytes(chunk_size=65536):
                    data.extend(chunk)
                    if len(data) > MAX_METADATA_BYTES:
                        raise InspectionError(f"Metadata exceeds {MAX_METADATA_BYTES} bytes: {url}")
    except httpx2.HTTPStatusError:
        raise
    except (httpx2.RequestError, OSError) as exc:
        raise InspectionError(f"Cannot read {url}: {exc}") from exc
    return data.decode("utf-8")


def normalize_version(value: str) -> str:
    value = value.removeprefix("v")
    if not re.fullmatch(r"\d+\.\d+\.\d+(?:(?:rc|a|b)\d+)?", value):
        raise InspectionError("vLLM version must be a release such as 0.34.0 (or v0.34.0)")
    return value


def static_value(node: ast.AST, bindings: JsonObject) -> Any:
    """Evaluate just literals and dictionary composition, never Python calls."""
    if isinstance(node, ast.Name) and node.id in bindings:
        return bindings[node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        values = [static_value(item, bindings) for item in node.elts]
        return tuple(values) if isinstance(node, ast.Tuple) else values
    if isinstance(node, ast.IfExp):
        # Registry values conditionally available with optional dependencies.
        # Record the class name, without checking/importing those dependencies.
        return static_value(node.body, bindings)
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq):
        return static_value(node.left, bindings) == static_value(node.comparators[0], bindings)
    if isinstance(node, ast.DictComp) and len(node.generators) == 1:
        generator = node.generators[0]
        iterator = generator.iter
        if not (
            isinstance(iterator, ast.Call)
            and isinstance(iterator.func, ast.Attribute)
            and iterator.func.attr == "items"
            and not iterator.args
            and not iterator.keywords
        ):
            raise ValueError("Unsupported registry comprehension")
        items = static_value(iterator.func.value, bindings).items()

        def bind(target: ast.AST, value: Any, scope: JsonObject) -> None:
            if isinstance(target, ast.Name):
                scope[target.id] = value
            elif isinstance(target, (ast.Tuple, ast.List)):
                for child, item in zip(target.elts, value):
                    bind(child, item, scope)
            else:
                raise ValueError("Unsupported comprehension binding")

        result = {}
        for key, value in items:
            scope = dict(bindings)
            bind(generator.target, (key, value), scope)
            if all(static_value(condition, scope) for condition in generator.ifs):
                result[static_value(node.key, scope)] = static_value(node.value, scope)
        return result
    if isinstance(node, ast.Dict):
        result = {}
        for key, value in zip(node.keys, node.values):
            if key is None:
                result.update(static_value(value, bindings))
            else:
                result[ast.literal_eval(key)] = static_value(value, bindings)
        return result
    if isinstance(node, ast.Call):
        function = node.func.value if isinstance(node.func, ast.Subscript) else node.func
        if not isinstance(function, ast.Name):
            raise ValueError("Unsupported call in registry")
        if function.id == "OrderedDict" and len(node.args) == 1:
            return dict(static_value(node.args[0], bindings))
        if function.id in ("LazyConfigDict", "OrderedDict") and not node.args:
            result = {}
            for keyword in node.keywords:
                if keyword.arg is None:
                    result.update(static_value(keyword.value, bindings))
                else:
                    # Only the keys of the config registry are needed.
                    result[keyword.arg] = None
            return result
    return ast.literal_eval(node)


def source_mapping(source: str, name: str) -> JsonObject:
    bindings = {}
    try:
        module = ast.parse(source)
        for node in module.body:
            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, ast.AnnAssign):
                targets, value = [node.target], node.value
            else:
                continue
            for target in targets:
                if isinstance(target, ast.Name) and value is not None:
                    try:
                        bindings[target.id] = static_value(value, bindings)
                    except (ValueError, TypeError, KeyError, SyntaxError):
                        continue
        result = bindings[name]
        if not isinstance(result, dict) or not result:
            raise ValueError("Expected a nonempty mapping")
        return result
    except (SyntaxError, KeyError, ValueError) as exc:
        raise InspectionError(
            f"Cannot statically read {name}; this release's source layout is unsupported"
        ) from exc


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)


def build_reference(version: str) -> ReleaseReference:
    version = normalize_version(version)
    base = f"https://raw.githubusercontent.com/vllm-project/vllm/v{version}"
    sources = {}

    def read(url: str) -> str:
        text = fetch_text(url)
        sources[url] = hashlib.sha256(text.encode()).hexdigest()
        return text

    try:
        registry = source_mapping(
            read(f"{base}/vllm/model_executor/models/registry.py"), "_VLLM_MODELS"
        )
        tools = source_mapping(
            read(f"{base}/vllm/tool_parsers/__init__.py"), "_TOOL_PARSERS_TO_REGISTER"
        )
        reasoning = source_mapping(
            read(f"{base}/vllm/reasoning/__init__.py"), "_REASONING_PARSERS_TO_REGISTER"
        )
        configs = source_mapping(
            read(f"{base}/vllm/transformers_utils/config.py"), "_CONFIG_REGISTRY"
        )
        requirements = read(f"{base}/requirements/common.txt")
        match = re.search(r"^transformers\s*(?:>=|==|~=)\s*([\d.]+)", requirements, re.M)
        if not match:
            raise InspectionError("Cannot determine the release's Transformers reference version")
        transformers_version = match[1]
        hf_base = f"https://raw.githubusercontent.com/huggingface/transformers/v{transformers_version}/src/transformers/models/auto"
        config_source = read(f"{hf_base}/configuration_auto.py")
        processor_source = read(f"{hf_base}/processing_auto.py")
        try:
            hf_configs = source_mapping(config_source, "CONFIG_MAPPING_NAMES")
            processors = source_mapping(processor_source, "PROCESSOR_MAPPING_NAMES")
        except InspectionError:
            # Transformers v5 moved generated mappings to a separate module.
            mappings = read(f"{hf_base}/auto_mappings.py")
            hf_configs = source_mapping(mappings, "CONFIG_MAPPING_NAMES")
            processors = source_mapping(mappings, "PROCESSOR_MAPPING_NAMES")
        tokenizers = source_mapping(
            read(f"{hf_base}/tokenization_auto.py"), "TOKENIZER_MAPPING_NAMES"
        )
    except httpx2.HTTPStatusError as exc:
        raise InspectionError(
            f"Cannot fetch vLLM {version} reference sources: HTTP {exc.response.status_code} at {exc.request.url}"
        ) from exc
    return {
        "version": version,
        "transformers_version": transformers_version,
        "architectures": {
            key: "transformers" if value[0] == "transformers" else "vllm"
            for key, value in sorted(registry.items())
        },
        "tool_parsers": sorted(tools),
        "reasoning_parsers": sorted(reasoning),
        "config_types": sorted(set(configs) | set(hf_configs)),
        "tokenizer_classes": sorted(
            {"PreTrainedTokenizerFast", "TokenizersBackend"}
            | set(_strings(list(tokenizers.values())))
            | {
                name + "Fast"
                for name in _strings(list(tokenizers.values()))
                if not name.endswith("Fast")
            }
        ),
        "processor_classes": sorted(set(_strings(list(processors.values())))),
        "sources": sources,
    }


def load_reference(version: str, offline: bool = False) -> ReleaseReference:
    version = normalize_version(version)
    cache_root = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "hf-vllm-recipe"
    cached = cache_root / f"vllm-{version}.json"
    if cached.is_file():
        try:
            reference: ReleaseReference = json.loads(cached.read_text(encoding="utf-8"))
            if reference["version"] != version or not reference["architectures"]:
                raise ValueError("Invalid release reference")
            for field in (
                "tool_parsers",
                "reasoning_parsers",
                "config_types",
                "tokenizer_classes",
                "processor_classes",
            ):
                if not isinstance(reference[field], list):
                    raise ValueError(f"Invalid {field}")
            return reference
        except (ValueError, KeyError) as exc:
            raise InspectionError(f"Invalid reference file {cached}: {exc}") from exc
    if offline:
        raise InspectionError(f"No cached reference for vLLM {version}; run once without --offline")
    reference = build_reference(version)
    try:
        cache_root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", dir=cache_root, delete=False, encoding="utf-8"
        ) as handle:
            temporary = Path(handle.name)
            json.dump(reference, handle, indent=2)
        temporary.replace(cached)
    except OSError:
        # A read-only cache must not prevent command generation.
        pass
    return reference
