"""Small, allowlisted tool set for the Omni Agent."""

from __future__ import annotations

import ast
import json
import math
import operator
import urllib.parse
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .safety import WriteApproval


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Keep the Brave API token bound to the configured Brave endpoint.
        return None


_SAFE_OPENER = build_opener(_NoRedirectHandler)


class ToolError(ValueError):
    """A safe, user-displayable tool failure."""


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any]], str]

    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    MAX_RESULT_CHARS = 24_000

    def __init__(self, specs: list[ToolSpec] | None = None) -> None:
        self._specs: dict[str, ToolSpec] = {}
        for spec in specs or []:
            self.register(spec)

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"Duplicate tool name: {spec.name}")
        self._specs[spec.name] = spec

    def openai_definitions(self) -> list[dict[str, Any]]:
        return [spec.definition() for spec in self._specs.values()]

    def names(self) -> list[str]:
        return list(self._specs)

    def execute(self, name: Any, arguments: Any) -> str:
        if not isinstance(name, str) or name not in self._specs:
            return "Tool error: unknown tool."
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                return "Tool error: arguments were not valid JSON."
        else:
            parsed = arguments
        if not isinstance(parsed, dict):
            return "Tool error: arguments must be a JSON object."
        try:
            result = self._specs[name].handler(parsed)
            if not isinstance(result, str):
                result = json.dumps(result, ensure_ascii=False)
        except ToolError as exc:
            result = f"Tool error: {exc}"
        except (OSError, UnicodeError) as exc:
            # Do not expose absolute host paths or stack traces to the model.
            result = f"Tool error: local file operation failed ({type(exc).__name__})."
        except Exception:
            result = "Tool error: the tool failed unexpectedly."
        if len(result) > self.MAX_RESULT_CHARS:
            result = result[: self.MAX_RESULT_CHARS] + "\n[tool output truncated]"
        return result


class WorkspaceTools:
    MAX_READ_BYTES = 100_000
    MAX_WRITE_CHARS = 100_000
    MAX_LIST_ITEMS = 100

    def __init__(self, root: Path | str, approval: WriteApproval) -> None:
        self.root = Path(root).expanduser().resolve()
        self.approval = approval
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, raw_path: Any) -> tuple[str, Path]:
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ToolError("path must be a non-empty relative path")
        value = raw_path.strip()
        windows_path = PureWindowsPath(value)
        if "\x00" in value or value.startswith(("/", "\\")) or windows_path.is_absolute() or windows_path.drive:
            raise ToolError("absolute paths are not allowed")
        posix_parts = Path(value).parts
        windows_parts = windows_path.parts
        if ".." in posix_parts or ".." in windows_parts:
            raise ToolError("parent-directory traversal is not allowed")
        candidate = (self.root / value).resolve()
        try:
            relative = candidate.relative_to(self.root)
        except ValueError as exc:
            raise ToolError("path resolves outside the workspace") from exc
        if candidate == self.root:
            return ".", candidate
        return relative.as_posix(), candidate

    def list_files(self, arguments: dict[str, Any]) -> str:
        relative, directory = self._resolve(arguments.get("path", "."))
        if not directory.exists() or not directory.is_dir():
            raise ToolError("directory does not exist")
        entries = sorted(directory.iterdir(), key=lambda path: path.name.casefold())
        result = []
        for entry in entries[: self.MAX_LIST_ITEMS]:
            try:
                is_dir = entry.is_dir()
                size = None if is_dir else entry.stat().st_size
            except OSError:
                is_dir, size = False, None
            result.append({"name": entry.name, "type": "directory" if is_dir else "file", "size_bytes": size})
        payload = {"path": relative, "entries": result, "truncated": len(entries) > self.MAX_LIST_ITEMS}
        return json.dumps(payload, ensure_ascii=False)

    def read_file(self, arguments: dict[str, Any]) -> str:
        relative, target = self._resolve(arguments.get("path"))
        if not target.exists() or not target.is_file():
            raise ToolError("file does not exist or is not a regular file")
        try:
            if target.stat().st_size > self.MAX_READ_BYTES:
                raise ToolError(f"file exceeds the {self.MAX_READ_BYTES}-byte read limit")
            content = target.read_bytes()
        except ToolError:
            raise
        except OSError as exc:
            raise ToolError("file could not be read") from exc
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ToolError("only UTF-8 text files can be read") from exc
        return json.dumps({"path": relative, "content": text}, ensure_ascii=False)

    def write_file(self, arguments: dict[str, Any]) -> str:
        relative, target = self._resolve(arguments.get("path"))
        content = arguments.get("content")
        if not isinstance(content, str):
            raise ToolError("content must be a string")
        if len(content) > self.MAX_WRITE_CHARS:
            raise ToolError(f"content exceeds the {self.MAX_WRITE_CHARS}-character write limit")
        if target.exists() and not target.is_file():
            raise ToolError("target exists and is not a regular file")
        overwriting = target.exists()
        if not self.approval.approve_write(relative, overwriting):
            raise ToolError("write was denied; no file was changed")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8", newline="")
        except OSError as exc:
            raise ToolError("file could not be written") from exc
        return json.dumps({"written": True, "path": relative, "characters": len(content)}, ensure_ascii=False)

    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="workspace_list",
                description="List files and folders directly inside a workspace directory. Paths are relative to the workspace.",
                parameters={
                    "type": "object",
                    "properties": {"path": {"type": "string", "description": "Relative directory; use '.' for the workspace root."}},
                    "required": [],
                    "additionalProperties": False,
                },
                handler=self.list_files,
            ),
            ToolSpec(
                name="workspace_read",
                description="Read a UTF-8 text file inside the workspace (maximum 100 KB).",
                parameters={
                    "type": "object",
                    "properties": {"path": {"type": "string", "description": "Relative file path inside the workspace."}},
                    "required": ["path"],
                    "additionalProperties": False,
                },
                handler=self.read_file,
            ),
            ToolSpec(
                name="workspace_write",
                description="Create or replace a UTF-8 text file inside the workspace. The CLI asks the user to approve each write unless explicitly run with --yes.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Relative file path inside the workspace."},
                        "content": {"type": "string", "description": "Complete UTF-8 file contents."},
                    },
                    "required": ["path", "content"],
                    "additionalProperties": False,
                },
                handler=self.write_file,
            ),
        ]


_ALLOWED_BINARY_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def safe_calculate(expression: Any) -> int | float:
    if not isinstance(expression, str) or not expression.strip():
        raise ToolError("expression must be a non-empty string")
    if len(expression) > 256:
        raise ToolError("expression is too long")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError("expression is not valid arithmetic") from exc
    if sum(1 for _ in ast.walk(tree)) > 100:
        raise ToolError("expression is too complex")

    def evaluate(node: ast.AST) -> int | float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = node.value
            if isinstance(value, float) and not math.isfinite(value):
                raise ToolError("non-finite numbers are not allowed")
            if abs(value) > 1e100:
                raise ToolError("number is outside the supported range")
            return value
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY_OPS:
            result = _ALLOWED_UNARY_OPS[type(node.op)](evaluate(node.operand))
            return check_result(result)
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINARY_OPS:
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Pow):
                if abs(right) > 12 or abs(left) > 1e20:
                    raise ToolError("exponentiation is limited to a small exponent and base")
            try:
                result = _ALLOWED_BINARY_OPS[type(node.op)](left, right)
            except (ArithmeticError, OverflowError, ValueError) as exc:
                raise ToolError("arithmetic operation failed") from exc
            return check_result(result)
        raise ToolError("only numeric literals and basic arithmetic operators are allowed")

    def check_result(value: Any) -> int | float:
        if type(value) not in (int, float):
            raise ToolError("unsupported arithmetic result")
        if isinstance(value, int) and value.bit_length() > 4_096:
            raise ToolError("integer result is too large")
        if isinstance(value, float) and not math.isfinite(value):
            raise ToolError("result is not finite")
        if abs(value) > 1e100:
            raise ToolError("result is outside the supported range")
        return value

    return evaluate(tree)


def _calculator_spec() -> ToolSpec:
    def calculate(arguments: dict[str, Any]) -> str:
        result = safe_calculate(arguments.get("expression"))
        return json.dumps({"expression": arguments["expression"], "result": result}, ensure_ascii=False)

    return ToolSpec(
        name="calculator",
        description="Evaluate a basic arithmetic expression safely. Supports numbers, parentheses, +, -, *, /, //, %, and small powers.",
        parameters={
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "A numeric arithmetic expression, for example '(19.5 * 3) / 2'."}},
            "required": ["expression"],
            "additionalProperties": False,
        },
        handler=calculate,
    )


class BraveSearch:
    ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str, timeout_seconds: float = 20.0) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def search(self, arguments: dict[str, Any]) -> str:
        query = arguments.get("query")
        count = arguments.get("count", 5)
        if not isinstance(query, str) or not query.strip():
            raise ToolError("query must be a non-empty string")
        if len(query) > 400:
            raise ToolError("query is limited to 400 characters")
        if type(count) is not int or not 1 <= count <= 5:
            raise ToolError("count must be an integer between 1 and 5")
        url = self.ENDPOINT + "?" + urllib.parse.urlencode({"q": query.strip(), "count": count})
        request = Request(
            url,
            headers={"Accept": "application/json", "X-Subscription-Token": self.api_key},
            method="GET",
        )
        try:
            with _SAFE_OPENER.open(request, timeout=self.timeout_seconds) as response:
                raw = response.read(2_000_001)
        except HTTPError as exc:
            raise ToolError(f"search provider returned HTTP {exc.code}") from exc
        except (URLError, TimeoutError) as exc:
            raise ToolError("search provider could not be reached") from exc
        if len(raw) > 2_000_000:
            raise ToolError("search response exceeded the 2 MB safety limit")
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ToolError("search provider returned invalid JSON") from exc
        items = data.get("web", {}).get("results", []) if isinstance(data, dict) else []
        results = []
        for item in items[:count] if isinstance(items, list) else []:
            if not isinstance(item, dict):
                continue
            results.append(
                {
                    "title": str(item.get("title", ""))[:300],
                    "url": str(item.get("url", ""))[:2_000],
                    "description": str(item.get("description", ""))[:1_200],
                }
            )
        return json.dumps({"query": query.strip(), "results": results}, ensure_ascii=False)

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="web_search",
            description="Search the public web with Brave Search. Search queries are sent to Brave; do not include secrets or private data.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "A concise public-web search query."},
                    "count": {"type": "integer", "description": "Number of results, from 1 to 5."},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            handler=self.search,
        )


def build_tool_registry(
    workspace: Path | str,
    approval: WriteApproval,
    *,
    brave_search_api_key: str | None = None,
) -> ToolRegistry:
    workspace_tools = WorkspaceTools(workspace, approval)
    specs = [_calculator_spec(), *workspace_tools.specs()]
    if brave_search_api_key:
        specs.append(BraveSearch(brave_search_api_key).spec())
    return ToolRegistry(specs)
