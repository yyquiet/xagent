"""Tests for tool_runner.py helper functions and main()."""

import base64
import json
from typing import Any, Mapping
from unittest.mock import patch

import cloudpickle
import pytest

from xagent.core.tools.adapters.vibe.sandboxed_tool.tool_runner import (
    _load_args,
    _load_init_params,
    _load_tool_class,
    _run_tool,
    main,
)


class _FakeTool:
    """Minimal fake tool for testing tool_runner."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def run_json_sync(self, args: Mapping[str, Any]) -> dict[str, Any]:
        return {"echo": args.get("msg", "")}

    async def run_json_async(self, args: Mapping[str, Any]) -> dict[str, Any]:
        return {"echo": args.get("msg", "")}


class TestLoadArgs:
    """Tests for _load_args()."""

    def test_roundtrip(self):
        """Base64-encoded JSON should decode back to original dict."""
        original = {"msg": "hello", "count": 42}
        b64 = base64.b64encode(json.dumps(original).encode()).decode()
        assert _load_args(b64) == original


class TestLoadInitParams:
    """Tests for _load_init_params()."""

    def test_none_returns_empty(self):
        """None input should return empty dict."""
        assert _load_init_params(None) == {}

    def test_roundtrip(self):
        """Cloudpickle-serialized params should deserialize correctly."""
        params = {"key": "value"}
        b64 = base64.b64encode(cloudpickle.dumps(params)).decode()
        assert _load_init_params(b64) == params


class TestLoadToolClass:
    """Tests for _load_tool_class()."""

    def test_valid_import(self):
        """Valid import path should resolve to the correct class."""
        cls = _load_tool_class(
            "tests.core.tools.adapters.sandboxed_tool.test_tool_runner:_FakeTool"
        )
        assert cls.__name__ == "_FakeTool"

    def test_invalid_module(self):
        """Non-existent module should raise ModuleNotFoundError."""
        with pytest.raises(ModuleNotFoundError):
            _load_tool_class("no.such.module:Cls")


class TestRunTool:
    """Tests for _run_tool()."""

    def test_sync(self):
        """Sync tool should return result directly."""
        tool = _FakeTool()
        assert _run_tool(tool, {"msg": "hi"}) == {"echo": "hi"}


class TestMain:
    """Tests for main() entrypoint."""

    def test_happy_path(self, tmp_path):
        """Successful execution should write result JSON to file."""
        result_file = str(tmp_path / "result.json")
        args_b64 = base64.b64encode(json.dumps({"msg": "ok"}).encode()).decode()
        tool_class = (
            "tests.core.tools.adapters.sandboxed_tool.test_tool_runner:_FakeTool"
        )
        argv = [
            "--tool-class",
            tool_class,
            "--args-b64",
            args_b64,
            "--result-file",
            result_file,
        ]
        with patch("sys.argv", ["tool_runner"] + argv):
            main()
        result = json.loads((tmp_path / "result.json").read_text())
        assert result == {"echo": "ok"}

    def test_bad_module_raises(self, tmp_path):
        """Invalid tool class should raise as Sandbox config error."""
        result_file = str(tmp_path / "result.json")
        args_b64 = base64.b64encode(b"{}").decode()
        argv = [
            "--tool-class",
            "no.such.module:Cls",
            "--args-b64",
            args_b64,
            "--result-file",
            result_file,
        ]
        with patch("sys.argv", ["tool_runner"] + argv):
            with pytest.raises(ModuleNotFoundError):
                main()
