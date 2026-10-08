"""A sandboxed virtual computer the agent can drive.

Deliberately strict: every file operation is confined to a single root
directory, paths are resolved and validated before use, and command execution
is limited to a small allow-list with timeouts. This is a safe local analogue
of the "own computer" that production agents use.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ActionResult:
    ok: bool
    output: str = ""
    error: str = ""


class SandboxError(Exception):
    pass


class VirtualComputer:
    def __init__(
        self,
        root: str,
        allowed_commands: Optional[List[str]] = None,
        max_output_chars: int = 4000,
        timeout_s: int = 10,
    ) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.allowed_commands = set(
            allowed_commands
            or ["ls", "cat", "head", "tail", "wc", "grep", "find", "echo", "python3", "pwd"]
        )
        self.max_output_chars = max_output_chars
        self.timeout_s = timeout_s
        self.clipboard: str = ""
        self._undo: List[Dict[str, Optional[str]]] = []

    # ---- path safety ----------------------------------------------------
    def _resolve(self, rel: str) -> Path:
        p = (self.root / rel).resolve()
        if self.root != p and self.root not in p.parents:
            raise SandboxError(f"path escapes sandbox root: {rel!r}")
        return p

    # ---- filesystem -----------------------------------------------------
    def write_file(self, rel: str, content: str) -> ActionResult:
        try:
            p = self._resolve(rel)
            prev = p.read_text() if p.exists() else None
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
            self._undo.append({"path": str(p), "prev": prev})
            return ActionResult(True, f"wrote {len(content)} chars to {rel}")
        except Exception as e:  # noqa: BLE001 - report any failure to the agent
            return ActionResult(False, error=str(e))

    def read_file(self, rel: str) -> ActionResult:
        try:
            p = self._resolve(rel)
            text = p.read_text()
            return ActionResult(True, text[: self.max_output_chars])
        except Exception as e:  # noqa: BLE001
            return ActionResult(False, error=str(e))

    def list_dir(self, rel: str = ".") -> ActionResult:
        try:
            p = self._resolve(rel)
            names = sorted(x.name + ("/" if x.is_dir() else "") for x in p.iterdir())
            return ActionResult(True, "\n".join(names))
        except Exception as e:  # noqa: BLE001
            return ActionResult(False, error=str(e))

    def undo_last(self) -> ActionResult:
        if not self._undo:
            return ActionResult(False, error="nothing to undo")
        entry = self._undo.pop()
        p = Path(entry["path"])
        if entry["prev"] is None:
            if p.exists():
                p.unlink()
        else:
            p.write_text(entry["prev"])
        return ActionResult(True, f"undid write to {p.name}")

    # ---- command execution ---------------------------------------------
    def run(self, command: str) -> ActionResult:
        try:
            parts = shlex.split(command)
        except ValueError as e:
            return ActionResult(False, error=f"bad command: {e}")
        if not parts:
            return ActionResult(False, error="empty command")
        if parts[0] not in self.allowed_commands:
            return ActionResult(
                False, error=f"command {parts[0]!r} not allowed (allow-list: {sorted(self.allowed_commands)})"
            )
        # never let shell metacharacters through
        if any(ch in command for ch in ";|&$`><\n"):
            return ActionResult(False, error="shell metacharacters are not permitted")
        try:
            proc = subprocess.run(
                parts,
                cwd=str(self.root),
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired:
            return ActionResult(False, error=f"timeout after {self.timeout_s}s")
        out = (proc.stdout + proc.stderr)[: self.max_output_chars]
        return ActionResult(proc.returncode == 0, out)

    # ---- clipboard / screenshot stub -----------------------------------
    def set_clipboard(self, text: str) -> ActionResult:
        self.clipboard = text
        return ActionResult(True, "clipboard set")

    def screenshot(self) -> ActionResult:
        # No display in a headless sandbox: return a textual "screen".
        listing = self.list_dir(".")
        return ActionResult(True, f"[screen] cwd={self.root}\n{listing.output}")

    # ---- JSON view used by the tool registry ---------------------------
    @staticmethod
    def _as_json(r: ActionResult) -> str:
        return json.dumps({"ok": r.ok, "output": r.output, "error": r.error})

    def write_file_json(self, path: str, content: str) -> str:
        return self._as_json(self.write_file(path, content))

    def read_file_json(self, path: str) -> str:
        return self._as_json(self.read_file(path))

    def list_dir_json(self, path: str = ".") -> str:
        return self._as_json(self.list_dir(path))

    def run_json(self, command: str) -> str:
        return self._as_json(self.run(command))

    def screenshot_json(self) -> str:
        return self._as_json(self.screenshot())
