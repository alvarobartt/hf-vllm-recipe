import contextlib
import io
import json
import os
import shlex
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx2

from hf_vllm_recipe.cli import main
from hf_vllm_recipe.inspection import (
    ModelSource,
    chat_templates,
    context_length,
    custom_code,
    infer_parsers,
)
from hf_vllm_recipe.reference import (
    InspectionError,
    build_reference,
    fetch_text,
    load_reference,
    normalize_version,
    source_mapping,
)
from tests.support import example_reference


class ModelFixture(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        loader = patch("hf_vllm_recipe.cli.load_reference", return_value=example_reference())
        loader.start()
        self.addCleanup(loader.stop)
        self.model = Path(self.directory.name) / "model with 'quotes'"
        self.model.mkdir()
        self.config = {
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "max_position_embeddings": 32768,
            "num_attention_heads": 32,
        }
        self.tokenizer = {
            "chat_template": "{{ tools }} <tool_call>{{ call }}</tool_call> <think>{{ reasoning }}</think>"
        }
        self.write("config.json", self.config)
        self.write("tokenizer_config.json", self.tokenizer)

    def write(self, name, value):
        target = self.model / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(value) if isinstance(value, (dict, list)) else value,
            encoding="utf-8",
        )

    def run_cli(self, *arguments):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = main([str(self.model), *arguments])
        return result, stdout.getvalue(), stderr.getvalue()


class CommandTests(ModelFixture):
    def test_default_command_and_shell_quoting(self):
        with patch.dict(sys.modules, {"hf_mem": None, "vllm": None, "transformers": None}):
            status, output, error = self.run_cli("--offline")
        self.assertEqual(status, 0, error)
        self.assertEqual(output.count("\n"), 1)
        self.assertEqual(error, "")
        self.assertEqual(
            shlex.split(output),
            [
                "vllm",
                "serve",
                str(self.model),
                "--max-model-len",
                "32768",
                "--enable-auto-tool-choice",
                "--tool-call-parser",
                "hermes",
                "--reasoning-parser",
                "qwen3",
            ],
        )

    def test_version_aliases(self):
        for option in ("--version", "-v"):
            status, output, error = self.run_cli(option, "v0.34.0", "--offline", "--verbose")
            self.assertEqual(status, 0, error)
            self.assertIn("vLLM 0.34.0", error)
            self.assertTrue(output.startswith("vllm serve"))

    def test_selected_reference_actually_changes_support(self):
        reference = dict(example_reference(), version="0.33.0", architectures={})
        with patch("hf_vllm_recipe.cli.load_reference", return_value=reference) as loader:
            status, output, error = self.run_cli("-v", "0.33.0")
        loader.assert_called_once_with("0.33.0", offline=False)
        self.assertEqual(status, 1)
        self.assertEqual(output, "")
        self.assertIn("0.33.0", error)

    def test_registered_transformers_model(self):
        self.write(
            "config.json",
            dict(self.config, architectures=["GemmaForCausalLM"], model_type="gemma"),
        )
        status, output, error = self.run_cli()
        self.assertEqual(status, 0, error)
        self.assertIn("--model-impl transformers", output)

    def test_unknown_architecture_fails(self):
        self.write("config.json", dict(self.config, architectures=["UnregisteredForCausalLM"]))
        status, output, error = self.run_cli()
        self.assertEqual(status, 1)
        self.assertEqual(output, "")
        self.assertIn("no matching", error)

    def test_explicit_transformers_fallback_is_provisional(self):
        self.write("config.json", dict(self.config, architectures=["UnregisteredForCausalLM"]))
        status, output, error = self.run_cli("--model-impl", "transformers")
        self.assertEqual(status, 0, error)
        self.assertIn("--model-impl transformers", output)
        self.assertIn("cannot be verified", error)

    def test_all_architectures_are_checked(self):
        self.write(
            "config.json",
            dict(self.config, architectures=["Unknown", "Qwen3ForCausalLM"]),
        )
        self.assertEqual(self.run_cli()[0], 0)

    def test_custom_config_requires_trust_even_with_native_model(self):
        self.write(
            "config.json",
            dict(
                self.config,
                model_type="my_custom_qwen",
                auto_map={
                    "AutoConfig": "configuration_custom.CustomConfig",
                    "AutoModelForCausalLM": "modeling_custom.CustomModel",
                },
            ),
        )
        self.write("configuration_custom.py", "raise RuntimeError('must never execute')")
        status, output, error = self.run_cli("--verbose")
        self.assertEqual(status, 0, error)
        self.assertIn("--trust-remote-code", output)
        self.assertIn("file present", error)
        self.assertNotIn("custom code: config.json: AutoModel", error)

    def test_native_auto_map_and_unreferenced_python_do_not_require_trust(self):
        self.write(
            "config.json",
            dict(
                self.config,
                auto_map={
                    "AutoConfig": "configuration_qwen.Qwen3Config",
                    "AutoModelForCausalLM": "modeling_qwen.Qwen3ForCausalLM",
                },
            ),
        )
        self.write("modeling_qwen.py", "raise RuntimeError('must never execute')")
        status, output, error = self.run_cli()
        self.assertEqual(status, 0, error)
        self.assertNotIn("trust-remote-code", output)

    def test_parser_overrides_and_disable(self):
        status, output, error = self.run_cli(
            "--tool-call-parser", "none", "--reasoning-parser", "deepseek_r1"
        )
        self.assertEqual(status, 0, error)
        self.assertNotIn("--enable-auto-tool-choice", output)
        self.assertNotIn("--tool-call-parser", output)
        self.assertIn("--reasoning-parser deepseek_r1", output)

    def test_unknown_explicit_parser_fails(self):
        status, output, error = self.run_cli("--tool-call-parser", "not_a_parser")
        self.assertEqual(status, 1)
        self.assertEqual(output, "")
        self.assertIn("Unknown tool-call parser", error)

    def test_unavailable_inferred_parser_is_omitted(self):
        reference = dict(example_reference(), reasoning_parsers=[])
        with patch("hf_vllm_recipe.cli.load_reference", return_value=reference):
            status, output, error = self.run_cli()
        self.assertEqual(status, 0)
        self.assertNotIn("--reasoning-parser", output)
        self.assertIn("unavailable", error)

    def test_missing_context_requires_override(self):
        del self.config["max_position_embeddings"]
        self.write("config.json", self.config)
        self.assertEqual(self.run_cli()[0], 1)
        self.assertEqual(self.run_cli("--max-model-len", "4096")[0], 0)

    def test_excessive_context_override_fails(self):
        status, output, error = self.run_cli("--max-model-len", "65536")
        self.assertEqual(status, 1)
        self.assertEqual(output, "")
        self.assertIn("exceeds", error)

    def test_malformed_metadata_fails_cleanly(self):
        self.write("config.json", "not json")
        status, output, error = self.run_cli()
        self.assertEqual(status, 1)
        self.assertEqual(output, "")
        self.assertIn("Invalid config.json", error)

    def test_missing_template_is_reported(self):
        self.write("tokenizer_config.json", {})
        status, output, error = self.run_cli()
        self.assertEqual(status, 0)
        self.assertNotIn("parser", output)
        self.assertIn("No chat template", error)

    def test_custom_template_is_read_and_forwarded(self):
        self.write("custom.jinja", "[TOOL_CALLS] {{ tools }}")
        status, output, error = self.run_cli("--chat-template", str(self.model / "custom.jinja"))
        self.assertEqual(status, 0, error)
        parts = shlex.split(output)
        self.assertEqual(parts[parts.index("--tool-call-parser") + 1], "mistral")
        self.assertIn(str(self.model / "custom.jinja"), parts)

    def test_native_mistral_formats_override_hf_architecture(self):
        self.write(
            "config.json",
            {"architectures": ["Mistral3ForConditionalGeneration"], "model_type": "mistral3"},
        )
        self.write(
            "params.json",
            {"max_position_embeddings": 131072, "vision_encoder": {"hidden_size": 1024}},
        )
        self.write("consolidated.safetensors", "")
        self.write("tekken.json", {})
        self.write("chat_template.jinja", "[TOOL_CALLS] {{ tools }}")
        status, output, error = self.run_cli("--verbose")
        self.assertEqual(status, 0, error)
        self.assertIn("PixtralForConditionalGeneration", error)
        self.assertIn(
            "--tokenizer-mode mistral --config-format mistral --load-format mistral", output
        )
        self.assertIn("--tool-call-parser mistral", output)
        self.assertIn("--max-model-len 131072", output)

    def test_params_only_mistral_checkpoint(self):
        (self.model / "config.json").unlink()
        self.write("params.json", {"max_position_embeddings": 32768, "n_heads": 32})
        self.write("consolidated-00001-of-00002.safetensors", "")
        self.write("tokenizer.model.v3", "")
        self.write("chat_template.jinja", "[TOOL_CALLS]")
        status, output, error = self.run_cli()
        self.assertEqual(status, 0, error)
        self.assertIn("--load-format mistral", output)
        self.assertIn("--max-model-len 32768", output)

    def test_hf_checkpoint_is_not_forced_to_mistral_format(self):
        self.write(
            "config.json",
            dict(self.config, model_type="mistral", architectures=["MistralForCausalLM"]),
        )
        self.write("params.json", {"max_position_embeddings": 8192})
        self.write("model.safetensors", "")
        self.write("tekken.json", {})
        status, output, error = self.run_cli()
        self.assertEqual(status, 0, error)
        self.assertNotIn("--load-format", output)
        self.assertNotIn("--config-format", output)
        self.assertIn("--max-model-len 32768", output)

    def test_deepseek_v4_default_context_and_native_encoder(self):
        self.write(
            "config.json",
            dict(
                self.config,
                model_type="deepseek_v4",
                architectures=["DeepseekV4ForCausalLM"],
                max_position_embeddings=1048576,
            ),
        )
        self.write("tokenizer_config.json", {})
        self.write(
            "encoding/encoding_dsv4.py",
            'raise RuntimeError("never execute")\ndsml_token="｜DSML｜"\ntool_calls_template="<{dsml_token}tool_calls> <think>"',
        )
        status, output, error = self.run_cli()
        self.assertEqual(status, 0, error)
        self.assertIn("--max-model-len 1048576", output)
        self.assertIn("--tokenizer-mode deepseek_v4", output)
        self.assertIn("--tool-call-parser deepseek_v4", output)
        self.assertIn("--reasoning-parser deepseek_v4", output)
        self.assertNotIn("--kv-cache-dtype", output)
        self.assertNotIn("--trust-remote-code", output)

    def test_native_tokenizer_mode_is_independent_of_parser_override(self):
        self.write(
            "config.json",
            dict(self.config, model_type="deepseek_v32", architectures=["DeepseekV32ForCausalLM"]),
        )
        self.write("encoding/encoding_dsv32.py", 'dsml_token="｜DSML｜"\nfunction_calls="<think>"')
        status, output, error = self.run_cli(
            "--tool-call-parser", "none", "--reasoning-parser", "none"
        )
        self.assertEqual(status, 0, error)
        self.assertIn("--tokenizer-mode deepseek_v32", output)
        self.assertNotIn("--tool-call-parser", output)

    def test_mistral_reasoning_delimiters(self):
        self.write("chat_template.jinja", "[THINK] {{ reasoning }} [/THINK] [TOOL_CALLS]")
        status, output, error = self.run_cli()
        self.assertEqual(status, 0, error)
        self.assertIn("--reasoning-parser mistral", output)
        self.assertIn("--tool-call-parser mistral", output)

    def test_native_mistral_renderer_takes_precedence_over_hf_template(self):
        self.write("params.json", {"max_position_embeddings": 32768})
        self.write("consolidated.safetensors", "")
        self.write("tekken.json", {})
        self.write(
            "tokenizer_config.json",
            dict(self.tokenizer, additional_special_tokens=["[TOOL_CALLS]", "[THINK]"]),
        )
        status, output, error = self.run_cli()
        self.assertEqual(status, 0, error)
        self.assertIn("--tool-call-parser mistral", output)
        self.assertIn("--reasoning-parser mistral", output)
        self.assertNotIn("--tool-call-parser hermes", output)

    def test_gpu_utilization_validation(self):
        self.assertEqual(self.run_cli("--gpu-memory-utilization", "nan")[0], 1)
        self.assertEqual(self.run_cli("--gpu-memory-utilization", "1.1")[0], 1)

    def test_hf_response_template(self):
        self.write(
            "tokenizer_config.json",
            dict(
                self.tokenizer,
                response_template={
                    "tool_calls": {"start": "<tool>"},
                    "thinking": {"start": "<think>"},
                },
            ),
        )
        status, output, error = self.run_cli()
        self.assertEqual(status, 0, error)
        self.assertIn("--tool-call-parser hf", output)
        self.assertIn("--reasoning-parser hf", output)
        self.assertEqual(self.run_cli("--reasoning-parser", "qwen3")[0], 1)


class MemoryTests(ModelFixture):
    def fake_memory(self, fits=True):
        report = {
            "total": 10 * 2**30,
            "fit": {"fits": fits, "max_model_len": 4096, "needed": 2},
            "warnings": [],
        }
        return Mock(return_value=SimpleNamespace(to_dict=lambda: report))

    def test_optional_dependency_error(self):
        with patch.dict(sys.modules, {"hf_mem": None}):
            status, output, error = self.run_cli("--gpu-memory", "1x96")
        self.assertEqual(status, 1)
        self.assertEqual(output, "")
        self.assertIn("hf-vllm-recipe[hf-mem]", error)

    def test_memory_call_matches_command(self):
        runner = self.fake_memory()
        with patch.dict(sys.modules, {"hf_mem": SimpleNamespace(run=runner)}):
            status, output, error = self.run_cli(
                "--gpu-memory",
                "2x80GiB",
                "--max-model-len",
                "8192",
                "--kv-cache-dtype",
                "fp8",
            )
        self.assertEqual(status, 0, error)
        self.assertIn("--tensor-parallel-size 2", output)
        self.assertIn("--kv-cache-dtype fp8", output)
        self.assertIn("10.00 GiB", error)
        kwargs = runner.call_args.kwargs
        self.assertEqual(kwargs["model_id"], str(self.model))
        self.assertEqual(kwargs["max_model_len"], 8192)
        self.assertEqual(kwargs["batch_size"], 1)
        self.assertEqual(kwargs["gpu_memory"], "2x80GiB")
        self.assertEqual(kwargs["engine"], "vllm")
        self.assertEqual(kwargs["format"], "safetensors")

    def test_failed_fit_never_emits_command(self):
        with patch.dict(sys.modules, {"hf_mem": SimpleNamespace(run=self.fake_memory(False))}):
            status, output, error = self.run_cli("--gpu-memory", "1x8")
        self.assertEqual(status, 1)
        self.assertEqual(output, "")
        self.assertIn("try --max-model-len 4096", error)

    def test_estimator_output_is_redirected(self):
        def noisy(**kwargs):
            print("estimator diagnostic")
            return self.fake_memory()()

        with patch.dict(sys.modules, {"hf_mem": SimpleNamespace(run=noisy)}):
            status, output, error = self.run_cli("--gpu-memory", "96GB")
        self.assertEqual(status, 0, error)
        self.assertEqual(output.count("\n"), 1)
        self.assertIn("estimator diagnostic", error)

    def test_parallelism_checks_attention_heads(self):
        with patch.dict(sys.modules, {"hf_mem": SimpleNamespace(run=self.fake_memory())}):
            status, output, error = self.run_cli("--gpu-memory", "3x96")
        self.assertEqual(status, 1)
        self.assertEqual(output, "")
        self.assertIn("not divisible", error)


class MetadataTests(ModelFixture):
    def test_legacy_json_chat_template(self):
        self.write("chat_template.json", {"chat_template": "[TOOL_CALLS]"})
        status, output, error = self.run_cli()
        self.assertEqual(status, 0, error)
        self.assertIn("--tool-call-parser mistral", output)
        self.assertNotIn("--reasoning-parser", output)

    def test_named_and_standalone_templates(self):
        self.write(
            "tokenizer_config.json",
            {
                "chat_template": [
                    {"name": "default", "template": "normal <think>"},
                    {"name": "tool_use", "template": "tool <tool_call>"},
                ]
            },
        )
        source = ModelSource(str(self.model))
        self.assertEqual(
            chat_templates(source, source.json("tokenizer_config.json")),
            ("tool <tool_call>", "normal <think>"),
        )
        self.write("chat_template.jinja", "standalone")
        self.write("chat_templates/tool_use.jinja", "standalone tools")
        source = ModelSource(str(self.model))
        self.assertEqual(
            chat_templates(source, source.json("tokenizer_config.json")),
            ("standalone tools", "standalone"),
        )

    def test_custom_tokenizer_reference(self):
        reference = example_reference()
        configs = {
            "tokenizer_config.json": {
                "tokenizer_class": "CustomTokenizer",
                "auto_map": {"AutoTokenizer": ["tokenization_custom.CustomTokenizer", None]},
            }
        }
        reasons = custom_code(configs, {"tokenization_custom.py"}, reference, "vllm")
        self.assertEqual(len(reasons), 1)
        self.assertIn("file present", reasons[0])
        configs["tokenizer_config.json"]["tokenizer_class"] = "Qwen2TokenizerFast"
        self.assertEqual(custom_code(configs, set(), reference, "vllm"), [])

    def test_context_rules(self):
        for config, expected in [
            (
                {
                    "text_config": {"max_position_embeddings": 8192},
                    "max_position_embeddings": 1024,
                },
                8192,
            ),
            ({"max_position_embeddings": 4096, "n_positions": 2048}, 2048),
            ({"max_position_embeddings": 4096, "model_max_length": 8192}, 8192),
            (
                {
                    "max_position_embeddings": 4096,
                    "rope_scaling": {"type": "linear", "factor": 2},
                },
                8192,
            ),
            (
                {
                    "max_position_embeddings": 131072,
                    "rope_scaling": {"rope_type": "llama3", "factor": 8},
                },
                131072,
            ),
            (
                {
                    "max_position_embeddings": 131072,
                    "rope_scaling": {"rope_type": "yarn", "factor": 4},
                },
                131072,
            ),
            (
                {
                    "max_position_embeddings": 131072,
                    "original_max_position_embeddings": 4096,
                    "rope_scaling": {"rope_type": "longrope"},
                },
                4096,
            ),
            (
                {
                    "model_type": "gemma3_text",
                    "max_position_embeddings": 131072,
                    "rope_scaling": {"type": "linear", "factor": 8},
                },
                131072,
            ),
        ]:
            with self.subTest(config=config):
                self.assertEqual(context_length(config, {}, []), expected)
        with self.assertRaises(InspectionError):
            context_length({}, {"model_max_length": 10**30}, [])

    def test_hub_commit_pinning_and_encoded_revision(self):
        def fetch(url, token):
            if "/api/models/" in url:
                self.assertIn("refs%2Fpr%2F4", url)
                return json.dumps({"sha": "deadbeef", "siblings": [{"rfilename": "config.json"}]})
            self.assertIn("/resolve/deadbeef/config.json", url)
            return json.dumps(self.config)

        with patch("hf_vllm_recipe.inspection.fetch_text", side_effect=fetch):
            source = ModelSource("org/model", "refs/pr/4")
            self.assertEqual(source.revision, "deadbeef")
            self.assertEqual(source.json("config.json"), self.config)
            self.assertIsNone(source.read("absent.json"))

    def test_hub_authentication_errors_not_hidden(self):
        request = httpx2.Request("GET", "https://huggingface.co/model")
        response = httpx2.Response(401, request=request)
        with patch(
            "hf_vllm_recipe.inspection.fetch_text",
            side_effect=httpx2.HTTPStatusError("denied", request=request, response=response),
        ):
            with self.assertRaisesRegex(InspectionError, "HF_TOKEN"):
                ModelSource("org/private")

    def test_parser_envelopes(self):
        for template, family, expected in [
            ("<tool_call> <function=f> <parameter=p>", "qwen3", "qwen3_xml"),
            ("<tool_call>{json}</tool_call>", "qwen3", "hermes"),
            ("[TOOL_CALLS]", "mistral", "mistral"),
            ("<|python_tag|>", "llama3", "llama3_json"),
            ("<|tool_call_begin|>", "kimi", "kimi_k2"),
            ("<｜tool▁calls▁begin｜>", "deepseek-v3.1", "deepseek_v31"),
            ("<tool_calls></tool_calls>", "jamba", "jamba"),
            ("<tool_call>{json}</tool_call>", "granitemoehybrid", "granite4"),
            ("<tool_call><function=example>", "granite", "granite_toolcall_parser"),
            ("<|tool_call>call:function", "gemma4", "gemma4"),
            ("<｜DSML｜function_calls>", "deepseek", "deepseek_v32"),
        ]:
            self.assertEqual(infer_parsers(template, "", family, {})[0], expected)
        self.assertEqual(
            infer_parsers("", "<think>", "deepseek-r1-distill-qwen3", {})[1],
            "deepseek_r1",
        )
        self.assertEqual(
            infer_parsers("no tool tokens", "plain template", "qwen3", {}), (None, None)
        )
        self.assertEqual(infer_parsers("", "<|channel>thought", "gemma4", {})[1], "gemma4")
        self.assertEqual(
            infer_parsers("<tool_call><function=example>", "<think>", "granite", {})[1],
            "granite_thinking_parser",
        )

    def test_deepseek_encoder_is_read_as_text(self):
        self.write("tokenizer_config.json", {})
        self.write(
            "config.json",
            dict(
                self.config,
                model_type="deepseek_v32",
                architectures=["DeepseekV32ForCausalLM"],
            ),
        )
        self.write(
            "encoding/encoding_dsv32.py",
            'raise RuntimeError("never executed")\ndsml_token="｜DSML｜"\ntemplate="<{dsml_token}function_calls> <think>"\nthinking_mode="chat"',
        )
        status, output, error = self.run_cli()
        self.assertEqual(status, 0, error)
        self.assertIn("--tool-call-parser deepseek_v32", output)
        self.assertIn("--reasoning-parser deepseek_v3", output)
        self.assertIn("--tokenizer-mode deepseek_v32", output)
        self.assertNotIn("--trust-remote-code", output)


class ReferenceTests(unittest.TestCase):
    def test_build_reference_from_tagged_sources(self):
        sources = {
            "registry.py": '_VLLM_MODELS = {"Qwen3ForCausalLM": ("qwen3", "Qwen3ForCausalLM"), "GemmaForCausalLM": ("transformers", "TransformersForCausalLM")}\nraise RuntimeError("never executed")',
            "tool_parsers/__init__.py": '_TOOL_PARSERS_TO_REGISTER = {"hermes": ("hermes_tool_parser", "Hermes2ProToolParser")}',
            "reasoning/__init__.py": '_REASONING_PARSERS_TO_REGISTER = {"qwen3": ("qwen3_engine_reasoning_parser", "Qwen3ParserReasoningAdapter")}',
            "config.py": '_CONFIG_REGISTRY = LazyConfigDict(custom="CustomConfig")',
            "common.txt": "transformers >= 5.16.1, < 5.23.0\n",
            "configuration_auto.py": "from .auto_mappings import CONFIG_MAPPING_NAMES",
            "processing_auto.py": "from .auto_mappings import PROCESSOR_MAPPING_NAMES",
            "auto_mappings.py": 'CONFIG_MAPPING_NAMES = {"qwen3": "Qwen3Config"}\nPROCESSOR_MAPPING_NAMES = {"qwen3": "Qwen3Processor"}',
            "tokenization_auto.py": 'TOKENIZER_MAPPING_NAMES = OrderedDict[str, str | None]([("qwen3", "Qwen2Tokenizer" if is_tokenizers_available() else None)])',
        }

        def fetch(url):
            for suffix, source in sources.items():
                if url.endswith(suffix):
                    return source
            self.fail(f"Unexpected source URL: {url}")

        with patch("hf_vllm_recipe.reference.fetch_text", side_effect=fetch) as fetcher:
            reference = build_reference("v0.34.0")
        self.assertEqual(reference["architectures"]["Qwen3ForCausalLM"], "vllm")
        self.assertEqual(reference["architectures"]["GemmaForCausalLM"], "transformers")
        self.assertIn("qwen3", reference["config_types"])
        self.assertIn("qwen3", reference["reasoning_parsers"])
        self.assertIn("hermes", reference["tool_parsers"])
        self.assertTrue(reference["sources"])
        self.assertIn("Qwen2TokenizerFast", reference["tokenizer_classes"])
        self.assertTrue(all(len(value) == 64 for value in reference["sources"].values()))
        self.assertTrue(any("/v0.34.0/" in call.args[0] for call in fetcher.call_args_list))
        self.assertTrue(any("/v5.16.1/" in call.args[0] for call in fetcher.call_args_list))

    def test_dynamic_reference_is_cached_for_offline_use(self):
        expected = example_reference()
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {"XDG_CACHE_HOME": directory}),
            patch("hf_vllm_recipe.reference.build_reference", return_value=expected) as builder,
        ):
            self.assertEqual(load_reference("0.34.0"), expected)
            builder.assert_called_once_with("0.34.0")
            builder.reset_mock()
            self.assertEqual(load_reference("0.34.0", offline=True), expected)
            builder.assert_not_called()

    def test_offline_requires_a_previously_fetched_reference(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {"XDG_CACHE_HOME": directory}),
            patch("hf_vllm_recipe.reference.build_reference") as builder,
        ):
            with self.assertRaisesRegex(InspectionError, "No cached reference"):
                load_reference("0.34.0", offline=True)
            builder.assert_not_called()

    def test_safe_source_reader(self):
        source = """
import module_that_does_not_exist
raise RuntimeError("never executed")
FIRST = {"LlamaForCausalLM": ("llama", "Llama")}
SECOND = {**{k: (mod, arch) for k, (mod, arch) in FIRST.items() if arch == "Llama"}}
RESULT = {**FIRST, **SECOND}
"""
        self.assertEqual(source_mapping(source, "RESULT"), {"LlamaForCausalLM": ("llama", "Llama")})
        with self.assertRaises(InspectionError):
            source_mapping('RESULT = __import__("os").system("false")', "RESULT")

    def test_invalid_version_and_unknown_offline_release(self):
        self.assertEqual(normalize_version("v0.34.0"), "0.34.0")
        with self.assertRaises(InspectionError):
            normalize_version("../../main")
        with self.assertRaises(InspectionError):
            load_reference("99.99.99", offline=True)

    def test_auth_removed_on_cross_host_redirect(self):
        def handle(request: httpx2.Request) -> httpx2.Response:
            if request.url.host == "huggingface.co":
                self.assertEqual(request.headers["Authorization"], "Bearer example")
                return httpx2.Response(302, headers={"Location": "https://cdn.example/config.json"})
            self.assertNotIn("Authorization", request.headers)
            return httpx2.Response(200, text="metadata")

        client = httpx2.Client(transport=httpx2.MockTransport(handle), follow_redirects=True)
        with patch("hf_vllm_recipe.reference.httpx2.Client", return_value=client):
            self.assertEqual(fetch_text("https://huggingface.co/org/model", "example"), "metadata")

    def test_http_metadata_size_limit(self):
        client = httpx2.Client(
            transport=httpx2.MockTransport(lambda request: httpx2.Response(200, content=b"x" * 128))
        )
        with (
            patch("hf_vllm_recipe.reference.httpx2.Client", return_value=client),
            patch("hf_vllm_recipe.reference.MAX_METADATA_BYTES", 64),
            self.assertRaisesRegex(InspectionError, "exceeds 64 bytes"),
        ):
            fetch_text("https://huggingface.co/config.json")


if __name__ == "__main__":
    unittest.main()
