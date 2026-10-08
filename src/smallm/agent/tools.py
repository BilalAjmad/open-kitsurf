"""Tool registry the agent can call during its reasoning loop."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Dict, List

from .computer import VirtualComputer


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, str]
    fn: Callable[..., str]

    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name}")
        return self._tools[name]

    def schemas(self) -> List[dict]:
        return [t.schema() for t in self._tools.values()]

    def call(self, name: str, args: dict) -> str:
        return self.get(name).fn(**args)


def _safe_calc(expression: str) -> str:
    """Evaluate a arithmetic expression with no names, calls, or attributes."""
    import ast
    import operator

    ops = {
        ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
        ast.USub: operator.neg, ast.UAdd: operator.pos,
    }

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](ev(node.operand))
        raise ValueError("unsupported expression")

    try:
        tree = ast.parse(expression, mode="eval")
        value = ev(tree)
    except Exception as e:  # noqa: BLE001
        import json
        return json.dumps({"ok": False, "output": "", "error": str(e)})
    import json
    return json.dumps({"ok": True, "output": str(value), "error": ""})


def default_registry(computer: VirtualComputer) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(Tool("calc", "Evaluate a integer arithmetic expression.",
                      {"expression": "str"}, _safe_calc))
    reg.register(Tool("write_file", "Write text to a file in the sandbox.",
                      {"path": "str", "content": "str"}, computer.write_file_json))
    reg.register(Tool("read_file", "Read a file from the sandbox.",
                      {"path": "str"}, computer.read_file_json))
    reg.register(Tool("list_dir", "List a directory.",
                      {"path": "str"}, computer.list_dir_json))
    reg.register(Tool("run", "Run an allow-listed shell command.",
                      {"command": "str"}, computer.run_json))
    reg.register(Tool("screenshot", "Capture the current screen state.",
                      {}, computer.screenshot_json))
    return reg
