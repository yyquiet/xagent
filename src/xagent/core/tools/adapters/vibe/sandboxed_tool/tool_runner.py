"""Execute a sandboxed tool from a stable Python entrypoint."""

from __future__ import annotations

import argparse
import asyncio
import base64
import importlib
import inspect
import json
from pathlib import Path
from typing import Any, cast

import cloudpickle  # type: ignore[import-untyped]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a sandboxed XAgent tool")
    parser.add_argument("--tool-class", required=True)
    parser.add_argument("--args-b64", required=True)
    parser.add_argument("--result-file", required=True)
    parser.add_argument("--init-params-b64")
    return parser.parse_args()


def _load_tool_class(import_path: str) -> type[Any]:
    module_path, class_name = import_path.rsplit(":", 1)
    module = importlib.import_module(module_path)
    return cast(type[Any], getattr(module, class_name))


def _load_args(args_b64: str) -> dict[str, Any]:
    args_json = base64.b64decode(args_b64).decode("utf-8")
    return cast(dict[str, Any], json.loads(args_json))


def _load_init_params(init_params_b64: str | None) -> dict[str, Any]:
    if not init_params_b64:
        return {}
    return cast(dict[str, Any], cloudpickle.loads(base64.b64decode(init_params_b64)))


def _run_tool(tool: Any, args: dict[str, Any]) -> Any:
    if inspect.iscoroutinefunction(tool.run_json_async):
        return asyncio.run(tool.run_json_async(args))
    return tool.run_json_sync(args)


def main() -> None:
    try:
        parsed = _parse_args()
        tool_class = _load_tool_class(parsed.tool_class)
        tool = tool_class(**_load_init_params(parsed.init_params_b64))
        tool_args = _load_args(parsed.args_b64)
    except Exception as e:
        print(f"Sandbox config error: {e}")
        raise

    result = _run_tool(tool, tool_args)

    result_path = Path(parsed.result_file)
    result_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
