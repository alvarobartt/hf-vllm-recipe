import contextlib
import io
import json
import shlex
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hf_vllm_recipe.cli import main
from hf_vllm_recipe.inspection import ModelSource
from tests.support import example_reference


class KitchenBaselineTests(unittest.TestCase):
    def test_generated_fields_against_validated_recipes(self):
        fixture = json.loads(
            (Path(__file__).parent / "fixtures" / "kitchen_baselines.json").read_text()
        )
        for record in fixture["models"]:
            with self.subTest(model=record["model"]), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for name, value in record["files"].items():
                    path = root / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(value if isinstance(value, str) else json.dumps(value))
                local = ModelSource(directory)
                # Use captured metadata through the real local-file reader,
                # while preserving the model ID and commit from the Hub.
                source = SimpleNamespace(
                    model=record["model"],
                    local=False,
                    revision=record["hub_revision"],
                    token=None,
                    files=local.files,
                    json=local.json,
                    read=local.read,
                )
                stdout, stderr = io.StringIO(), io.StringIO()
                with (
                    patch("hf_vllm_recipe.cli.ModelSource", return_value=source),
                    patch("hf_vllm_recipe.cli.load_reference", return_value=example_reference()),
                    contextlib.redirect_stdout(stdout),
                    contextlib.redirect_stderr(stderr),
                ):
                    status = main([record["model"], *record["arguments"]])
                self.assertEqual(status, 0, stderr.getvalue())
                self.assertEqual(stdout.getvalue().count("\n"), 1)
                tokens = shlex.split(stdout.getvalue())
                self.assertEqual(tokens[:3], ["vllm", "serve", record["model"]])
                generated: dict[str, str | bool] = {"model": tokens[2]}
                index = 3
                while index < len(tokens):
                    flag = tokens[index].removeprefix("--")
                    if index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
                        generated[flag] = tokens[index + 1]
                        index += 2
                    else:
                        generated[flag] = True
                        index += 1
                expected = record["expected"]
                # Recipe GPU settings, kernels, scheduling, cache tuning, etc.
                # are intentionally absent from the fixture and the command.
                self.assertLessEqual(set(generated), set(expected) | {"revision"})
                for flag, value in expected.items():
                    with self.subTest(flag=flag):
                        wanted = True if value is None else str(value)
                        if flag == "tool-call-parser":
                            wanted = fixture["parser_aliases"].get(wanted, wanted)
                        self.assertEqual(generated.get(flag), wanted)


if __name__ == "__main__":
    unittest.main()
