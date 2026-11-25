import argparse
import contextlib
import re
import shlex
import sys
from collections.abc import Sequence
from pathlib import Path

from .formats import model_config, native_tool_parser, tokenizer_mode
from .inspection import (
    ModelSource,
    architecture,
    chat_templates,
    context_length,
    custom_code,
    infer_parsers,
    text_config,
)
from .reference import (
    DEFAULT_VERSION,
    InspectionError,
    load_reference,
    normalize_version,
)


def positive(value: str) -> int:
    try:
        number = int(value)
        if number > 0:
            return number
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("must be a positive integer")


def gpu_memory(value: str) -> str:
    match = re.fullmatch(r"(?:(\d+)x)?(\d+(?:\.\d+)?)(GiB|GB)?", value.strip(), re.I)
    if not match or int(match[1] or 1) <= 0 or float(match[2]) <= 0:
        raise argparse.ArgumentTypeError("expected memory per GPU, e.g. 1x96, 2x80GiB, or 96GB")
    return value.strip()


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hf-vllm-recipe",
        description="Generate a vLLM serve command from model metadata.",
    )
    parser.add_argument("model", help="Hub ID (org/model) or local model directory")
    parser.add_argument(
        "--version",
        "-v",
        default=DEFAULT_VERSION,
        metavar="VLLM_VERSION",
        help=f"vLLM reference release, defaults to {DEFAULT_VERSION}; does not install vLLM",
    )
    parser.add_argument(
        "--revision",
        help="Hub branch, tag or commit; resolved to a commit for reproducibility",
    )
    parser.add_argument(
        "--max-model-len",
        type=positive,
        help="Override the context length read from model configuration",
    )
    parser.add_argument("--model-impl", choices=("auto", "vllm", "transformers"), default="auto")
    parser.add_argument(
        "--tool-call-parser",
        metavar="NAME",
        help="Override parser inference; 'none' disables it",
    )
    parser.add_argument(
        "--reasoning-parser",
        metavar="NAME",
        help="Override parser inference; 'none' disables it",
    )
    parser.add_argument(
        "--chat-template", type=Path, help="Inspect and use this local Jinja template"
    )
    parser.add_argument(
        "--gpu-memory",
        type=gpu_memory,
        help="Run hf-mem, e.g. 1x96 or 2x80GiB; bare size means one GPU",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="GPU budget fraction, defaults to 0.90",
    )
    parser.add_argument(
        "--kv-cache-dtype",
        default="auto",
        help="KV cache dtype for both estimation and serving",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use a local model and cached release reference",
    )
    parser.add_argument("--verbose", action="store_true", help="Explain decisions on stderr")
    return parser


def select_parser(
    override: str | None,
    inferred: str | None,
    available: Sequence[str],
    kind: str,
    warnings: list[str],
) -> str | None:
    if override == "none":
        return None
    selected = override or inferred
    if selected and selected not in available:
        if override:
            raise InspectionError(
                f"Unknown {kind} parser {selected!r} for the selected release; available: {', '.join(available)}"
            )
        if selected == "qwen3_xml" and "qwen3_coder" in available:
            return "qwen3_coder"
        warnings.append(
            f"Inferred {kind} parser {selected!r} is unavailable in this vLLM release; use --{kind}-parser to override"
        )
        return None
    return selected


def estimate_memory(source: ModelSource, args: argparse.Namespace, length: int) -> int:
    # Import only when explicitly requested.
    try:
        from hf_mem import run
    except ImportError as exc:
        raise InspectionError(
            "Memory estimation requires the hf-mem extra. Run: uvx 'hf-vllm-recipe[hf-mem]' MODEL --gpu-memory SIZE"
        ) from exc
    memory = args.gpu_memory if "x" in args.gpu_memory.lower() else f"1x{args.gpu_memory}"
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = run(
                model_id=source.model,
                revision=source.revision,
                hf_token=source.token,
                max_model_len=length,
                batch_size=1,
                kv_cache_dtype=args.kv_cache_dtype,
                gpu_memory=memory,
                gpu_memory_utilization=args.gpu_memory_utilization,
                engine="vllm",
                format="safetensors",
            )
        report = result.to_dict()
    except Exception as exc:
        raise InspectionError(f"hf-mem estimation failed: {exc}") from exc
    total = report.get("total")
    if not isinstance(total, (int, float)) or total < 0:
        raise InspectionError("hf-mem returned no single-model estimate")
    print(
        f"hf-mem: {total / 2**30:.2f} GiB estimated for {length} tokens, one sequence, {memory}, "
        f"utilization={args.gpu_memory_utilization:g}",
        file=sys.stderr,
    )
    for warning in report.get("warnings", []):
        print(f"hf-mem: warning: {warning}", file=sys.stderr)
    fit = report.get("fit")
    if not isinstance(fit, dict) or not isinstance(fit.get("fits"), bool):
        raise InspectionError("hf-mem did not return a GPU-fit assessment")
    if not fit["fits"]:
        suggestions = []
        if fit.get("max_model_len", 0) > 0:
            suggestions.append(f"try --max-model-len {fit['max_model_len']}")
        if fit.get("needed"):
            suggestions.append(f"hf-mem estimates {fit['needed']} GPUs for the requested context")
        raise InspectionError(
            f"Model is estimated not to fit {memory} at {length} tokens"
            + ("; " + "; ".join(suggestions) if suggestions else "")
        )
    count = int(memory.lower().split("x", 1)[0])
    config, _ = model_config(source)
    heads = text_config(config).get("num_attention_heads")
    if isinstance(heads, int) and heads % count:
        raise InspectionError(
            f"{heads} attention heads are not divisible by tensor parallel size {count}; choose a different GPU count"
        )
    return count


def command(args: argparse.Namespace) -> str:
    if not 0 < args.gpu_memory_utilization <= 1:
        raise InspectionError("--gpu-memory-utilization must be in (0, 1]")
    reference = load_reference(normalize_version(args.version), offline=args.offline)
    source = ModelSource(args.model, args.revision, args.offline)
    config, formats = model_config(source)
    tokenizer = source.json("tokenizer_config.json")
    warnings = []
    name, implementation = architecture(config, reference, args.model_impl)
    if formats.get("config-format") == "mistral" and text_config(config) is not config:
        architecture(text_config(config), reference, args.model_impl)
    if args.model_impl == "transformers" and name not in reference["architectures"]:
        warnings.append(
            "Transformers backend was explicitly requested; runtime compatibility cannot be verified without loading the model"
        )
    length = args.max_model_len or context_length(config, tokenizer, warnings)
    if args.max_model_len:
        try:
            configured = context_length(config, tokenizer, [])
        except InspectionError:
            configured = None
        if configured is not None and length > configured:
            raise InspectionError(
                f"--max-model-len {length} exceeds the configured context length {configured}"
            )
    configs = {"config.json": config, "tokenizer_config.json": tokenizer}
    for filename in ("processor_config.json", "preprocessor_config.json"):
        configs[filename] = source.json(filename)
    reasons = custom_code(configs, source.files, reference, implementation)
    if args.chat_template:
        tool_template = reasoning_template = args.chat_template.read_text(encoding="utf-8")
    else:
        tool_template, reasoning_template = chat_templates(source, tokenizer)
    family = " ".join((source.model, name, str(config.get("model_type", "")))).lower()
    inferred_tool, inferred_reasoning = infer_parsers(
        tool_template, reasoning_template, family, tokenizer
    )
    inferred_tool = native_tool_parser(source, tokenizer, formats) or inferred_tool
    if formats.get("tokenizer-mode") == "mistral" and inferred_reasoning:
        inferred_reasoning = "mistral"
    tool = select_parser(
        args.tool_call_parser,
        inferred_tool,
        reference["tool_parsers"],
        "tool-call",
        warnings,
    )
    reasoning = select_parser(
        args.reasoning_parser,
        inferred_reasoning,
        reference["reasoning_parsers"],
        "reasoning",
        warnings,
    )
    if (tool == "hf" and reasoning not in (None, "hf")) or (
        reasoning == "hf" and tool not in (None, "hf")
    ):
        raise InspectionError(
            "The hf parser cannot be combined with a model-specific tool or reasoning parser"
        )
    if tool == "hf" or reasoning == "hf":
        response = tokenizer.get("response_template") or {}
        if (
            not isinstance(response, dict)
            or (tool == "hf" and not response.get("tool_calls"))
            or (reasoning == "hf" and not response.get("thinking"))
        ):
            raise InspectionError(
                "The hf parser requires matching tool_calls/thinking response_template metadata"
            )
    if (
        not tool_template
        and not reasoning_template
        and not formats.get("tokenizer-mode") == "mistral"
    ):
        warnings.append("No chat template found; the chat endpoint may require --chat-template")
    elif (
        not tool
        and args.tool_call_parser != "none"
        and re.search(r"\btools\b|tool_calls", tool_template)
    ):
        warnings.append(
            "Template mentions tools but its output format was not recognized; use --tool-call-parser to override"
        )
    if reasons:
        warnings.append(
            "Adding --trust-remote-code for referenced custom classes; the generated command will allow model-provided code to execute"
        )
    parts = ["vllm", "serve", source.model, "--max-model-len", str(length)]
    if not source.local:
        if source.revision is None:
            raise InspectionError("Hub model has no resolved revision")
        parts += ["--revision", source.revision]
    if implementation == "transformers" or args.model_impl == "vllm":
        parts += ["--model-impl", implementation]
    if mode := tokenizer_mode(config):
        formats.setdefault("tokenizer-mode", mode)
    for flag, value in formats.items():
        parts += [f"--{flag}", value]
    if reasons:
        parts.append("--trust-remote-code")
    if args.chat_template:
        parts += ["--chat-template", str(args.chat_template.resolve())]
    if tool:
        parts += ["--enable-auto-tool-choice", "--tool-call-parser", tool]
    if reasoning:
        parts += ["--reasoning-parser", reasoning]
    if args.kv_cache_dtype != "auto":
        parts += ["--kv-cache-dtype", args.kv_cache_dtype]
    if args.gpu_memory or args.gpu_memory_utilization != 0.90:
        parts += ["--gpu-memory-utilization", str(args.gpu_memory_utilization)]
    for warning in warnings:
        print(f"hf-vllm-recipe: warning: {warning}", file=sys.stderr)
    if args.verbose:
        print(
            f"vLLM {reference['version']}: {name}, implementation={implementation}, context={length}, "
            f"Python files={sum(path.endswith('.py') for path in source.files)}",
            file=sys.stderr,
        )
        for reason in reasons:
            print(f"custom code: {reason}", file=sys.stderr)
    if args.gpu_memory:
        count = estimate_memory(source, args, length)
        if count > 1:
            parts += ["--tensor-parallel-size", str(count)]
    return shlex.join(parts)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argument_parser()
    args = parser.parse_args(argv)
    try:
        result = command(args)
    except (InspectionError, OSError, ValueError) as exc:
        print(f"hf-vllm-recipe: error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    print(result)
    return 0
