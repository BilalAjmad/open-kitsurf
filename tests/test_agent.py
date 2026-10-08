import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import tempfile
from smallm.agent import (
    VirtualComputer, default_registry, AgentLoop, ScriptedBrain,
    Verifier, parse_decision, validate_english, SandboxError,
)


def test_sandbox_blocks_escape():
    with tempfile.TemporaryDirectory() as d:
        c = VirtualComputer(d)
        r = c.write_file("../evil.txt", "x")
        assert not r.ok and "escapes" in r.error


def test_sandbox_command_allowlist():
    with tempfile.TemporaryDirectory() as d:
        c = VirtualComputer(d)
        c.write_file("a.txt", "hello")
        assert c.run("cat a.txt").ok
        assert not c.run("rm a.txt").ok
        assert not c.run("cat a.txt; rm a.txt").ok


def test_sandbox_undo():
    with tempfile.TemporaryDirectory() as d:
        c = VirtualComputer(d)
        c.write_file("a.txt", "hello")
        c.undo_last()
        assert not (os.path.exists(os.path.join(d, "a.txt")))


def test_agent_uses_tool_then_answers():
    with tempfile.TemporaryDirectory() as d:
        c = VirtualComputer(d)
        reg = default_registry(c)
        brain = ScriptedBrain([
            {"thought": "I write the file first.",
             "tool": "write_file", "args": {"path": "a.txt", "content": "hello"}},
            {"thought": "I read it to check.",
             "tool": "read_file", "args": {"path": "a.txt"}},
            {"thought": "The content matches.",
             "answer": "The file contains hello."},
        ])
        loop = AgentLoop(brain, reg, Verifier(required_keywords=["hello"]))
        res = loop.run("Create a.txt with hello and verify it.")
        assert res.passed
        assert "hello" in res.answer
        kinds = [s.kind for s in res.steps]
        assert "action" in kinds and "observation" in kinds and "verification" in kinds


def test_agent_self_corrects_on_verification_failure():
    with tempfile.TemporaryDirectory() as d:
        c = VirtualComputer(d)
        reg = default_registry(c)
        brain = ScriptedBrain([
            {"thought": "first try", "answer": "This answer is wrong."},   # missing keyword
            {"thought": "fix it", "answer": "The needed word is hello."},  # now passes
        ])
        loop = AgentLoop(brain, reg, Verifier(required_keywords=["hello"]))
        res = loop.run("answer with hello")
        assert res.passed
        assert res.iterations == 2


def test_strict_english_validator_and_corrector():
    bad = "I don't think it's fine! Also this sentence is very very very very very very very very very very long;"
    rep = validate_english(bad)
    assert not rep.compliant
    assert "do not" in rep.corrected_text
    assert rep.score < 1.0
    good = "I do not think it is fine. The sentence is short."
    assert validate_english(good).compliant


def test_parse_decision_tool_and_answer():
    t = ' thinkingI should write.</think><tool_call>{"name": "write_file", "args": {"path": "x"}}</tool_call>'
    d = parse_decision(t)
    assert d["tool"] == "write_file" and d["args"]["path"] == "x"
    a = " thinkingdone</think><verify>checked</verify>The result is 4."
    d2 = parse_decision(a)
    assert d2.get("answer", "").strip() == "The result is 4."


def test_loop_stops_at_max_iterations():
    with tempfile.TemporaryDirectory() as d:
        c = VirtualComputer(d)
        reg = default_registry(c)
        brain = ScriptedBrain([
            {"thought": "loop", "tool": "list_dir", "args": {"path": "."}} for _ in range(50)
        ])
        loop = AgentLoop(brain, reg, max_iterations=3)
        res = loop.run("loop forever")
        assert not res.passed and res.iterations == 3
