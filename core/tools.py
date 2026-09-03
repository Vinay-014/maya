"""Safe built-in tools and the agent tool registry."""

from __future__ import annotations

import ast
import json
import math
import operator
import os
import platform
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


class ToolError(Exception):
    """Raised when a tool request is invalid or unsafe."""


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., dict[str, Any]]

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """Extensible registry for validated, executable agent tools."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool

    def list_tools(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolError(f"Unknown tool: {name}")
        try:
            return tool.handler(**arguments)
        except TypeError as exc:
            raise ToolError(f"Invalid arguments for {name}: {exc}") from exc


def get_system_status() -> dict[str, Any]:
    """Return non-sensitive local runtime and resource information."""
    memory: dict[str, Any] = {"available": False}
    try:
        import psutil

        memory = {
            "available": True,
            "percent_used": psutil.virtual_memory().percent,
            "available_mb": round(psutil.virtual_memory().available / 1048576, 1),
        }
        cpu_percent = psutil.cpu_percent(interval=0.1)
    except ImportError:
        cpu_percent = None
    return {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "cpu_count": os.cpu_count(),
        "cpu_percent": cpu_percent,
        "memory": memory,
        "cwd": str(Path.cwd()),
    }


_ALLOWED_BINARY_OPERATORS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_ALLOWED_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def calculate_math(expression: str) -> dict[str, Any]:
    """Safely evaluate arithmetic without executing Python code."""
    normalized = _extract_math_expression(expression)
    if len(normalized) > 200:
        raise ToolError("Expression is too long")

    def evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            if not math.isfinite(float(node.value)):
                raise ToolError("Non-finite number")
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINARY_OPERATORS:
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, (ast.Div, ast.Mod, ast.FloorDiv)) and right == 0:
                raise ToolError("Division by zero")
            result = _ALLOWED_BINARY_OPERATORS[type(node.op)](left, right)
            if abs(result) > 1e15:
                raise ToolError("Result is too large")
            return float(result)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY_OPERATORS:
            return float(_ALLOWED_UNARY_OPERATORS[type(node.op)](evaluate(node.operand)))
        raise ToolError("Only basic arithmetic is allowed")

    try:
        result = evaluate(ast.parse(normalized, mode="eval").body)
    except (SyntaxError, ValueError, TypeError) as exc:
        raise ToolError("Invalid arithmetic expression") from exc
    return {"expression": normalized, "result": result}


def _extract_math_expression(value: str) -> str:
    """Normalize calculator symbols and isolate arithmetic from conversational text."""
    import re

    normalized = value.strip().replace("×", "*").replace("÷", "/")
    normalized = re.sub(r"(?<=\d)\s*[xX]\s*(?=\d|\()", "*", normalized)
    normalized = re.sub(r"(?<=\d),(?=\d)", "", normalized)
    match = re.search(r"(?=[0-9])[-+*/%().\d\s]+", normalized)
    if not match:
        raise ToolError("Invalid arithmetic expression")
    candidate = match.group(0).strip()
    candidate = re.sub(r"\s+", " ", candidate)
    if not re.search(r"[+*/%()-]", candidate) and not re.fullmatch(r"\d+(?:\.\d+)?", candidate):
        raise ToolError("Invalid arithmetic expression")
    return candidate


def search_local_notes(query: str) -> dict[str, Any]:
    """Search workspace markdown and text notes, excluding runtime/vendor folders."""
    needle = query.strip().lower()
    if not needle:
        raise ToolError("Query cannot be empty")
    results: list[dict[str, Any]] = []
    excluded = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "sessions"}
    for path in Path.cwd().rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
            continue
        if any(part in excluded for part in path.parts):
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        hits = [{"line": index, "text": line.strip()[:240]} for index, line in enumerate(lines, 1) if needle in line.lower()]
        if hits:
            results.append({"path": str(path.relative_to(Path.cwd())), "matches": hits[:5]})
        if len(results) >= 20:
            break
    return {"query": query, "results": results}


_SAFE_SHELL_COMMANDS = {"dir", "git", "pwd", "where", "whoami"}
_READ_ONLY_GIT = {"status", "log", "diff", "branch"}


def execute_shell_cmd(command: str) -> dict[str, Any]:
    """Execute a narrowly allowlisted read-only command in the workspace."""
    try:
        args = shlex.split(command, posix=False)
    except ValueError as exc:
        raise ToolError("Invalid shell command") from exc
    if not args or args[0].lower() not in _SAFE_SHELL_COMMANDS:
        raise ToolError("Command is not allowlisted")
    executable = args[0].lower()
    if executable == "git" and (len(args) < 2 or args[1].lower() not in _READ_ONLY_GIT):
        raise ToolError("Only read-only git commands are allowed")
    if any(token in command for token in ("&", "|", ";", ">", "<", "`", "$")):
        raise ToolError("Shell operators are not allowed")
    if executable == "dir":
        args = ["cmd.exe", "/c", "dir", *args[1:]]
    completed = subprocess.run(args, cwd=Path.cwd(), capture_output=True, text=True, timeout=10, shell=False)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-1000:],
    }


def default_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ToolDefinition("get_system_status", "Inspect local OS, CPU, memory, and Python runtime details.", {"type": "object", "properties": {}}, lambda: get_system_status()))
    registry.register(ToolDefinition("calculate_math", "Safely calculate a basic arithmetic or financial expression.", {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}, calculate_math))
    registry.register(ToolDefinition("search_local_notes", "Search workspace Markdown and text notes for a query.", {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}, search_local_notes))
    registry.register(ToolDefinition("execute_shell_cmd", "Run an allowlisted read-only workspace command such as dir or git status.", {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}, execute_shell_cmd))
    return registry


def execute_tool_json(registry: ToolRegistry, name: str, arguments: dict[str, Any]) -> str:
    try:
        return json.dumps(registry.execute(name, arguments), ensure_ascii=True)
    except ToolError as exc:
        return json.dumps({"error": str(exc)})
