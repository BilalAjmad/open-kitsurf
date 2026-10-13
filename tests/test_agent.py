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


def test_loop_rejects_answer_that_contradicts_tool_result():
    with tempfile.TemporaryDirectory() as d:
        c = VirtualComputer(d)
        reg = default_registry(c)
        brain = ScriptedBrain([
            {"thought": "I compute.", "tool": "calc", "args": {"expression": "2+3"}},
            {"thought": "wrong on purpose", "answer": "The answer is 9."},   # contradicts 5
            {"thought": "now correct", "answer": "The answer is 5."},        # matches
        ])
        loop = AgentLoop(brain, reg)
        res = loop.run("Compute 2+3.")
        assert res.passed
        assert "5" in res.answer
        assert res.iterations == 3


def test_loop_fails_when_answer_never_matches_observation():
    with tempfile.TemporaryDirectory() as d:
        c = VirtualComputer(d)
        reg = default_registry(c)
        brain = ScriptedBrain([
            {"thought": "I compute.", "tool": "calc", "args": {"expression": "2+3"}},
            {"thought": "nope", "answer": "The answer is 9."},
        ])
        loop = AgentLoop(brain, reg, max_verify_attempts=1)
        res = loop.run("Compute 2+3.")
        assert not res.passed


def test_loop_rejects_answer_with_leaked_control_token():
    with tempfile.TemporaryDirectory() as d:
        c = VirtualComputer(d)
        reg = default_registry(c)
        brain = ScriptedBrain([
            {"thought": "bad", "answer": "The answer is 5.</tool_call> garbage"},
            {"thought": "good", "answer": "The answer is 5."},
        ])
        loop = AgentLoop(brain, reg)
        res = loop.run("Compute 2+3.")
        assert res.passed
        assert res.iterations == 2
        assert "</tool_call>" not in res.answer


def test_loop_rejects_json_answer_and_recovers():
    with tempfile.TemporaryDirectory() as d:
        c = VirtualComputer(d)
        reg = default_registry(c)
        brain = ScriptedBrain([
            {"thought": "run", "tool": "calc", "args": {"expression": "2+3"}},
            {"thought": "dump", "answer": '{"output": "5"}'},
            {"thought": "ok", "answer": "The answer is 5."},
        ])
        loop = AgentLoop(brain, reg)
        res = loop.run("Compute 2+3.")
        assert res.passed and res.answer == "The answer is 5."


def test_loop_ignores_repeated_tool_call():
    with tempfile.TemporaryDirectory() as d:
        c = VirtualComputer(d)
        reg = default_registry(c)
        brain = ScriptedBrain([
            {"thought": "run", "tool": "calc", "args": {"expression": "2+3"}},
            {"thought": "again", "tool": "calc", "args": {"expression": "2+3"}},
            {"thought": "ok", "answer": "The answer is 5."},
        ])
        loop = AgentLoop(brain, reg)
        res = loop.run("Compute 2+3.")
        assert res.passed
        actions = [s for s in res.steps if s.kind == "action"]
        assert len(actions) == 1


def test_loop_requires_file_tool_before_file_answer():
    with tempfile.TemporaryDirectory() as d:
        c = VirtualComputer(d)
        reg = default_registry(c)
        brain = ScriptedBrain([
            {"thought": "lie", "answer": "The file was created."},
            {"thought": "do it", "tool": "write_file",
             "args": {"path": "note.txt", "content": "hello"}},
            {"thought": "ok", "answer": "The note.txt file now holds hello."},
        ])
        loop = AgentLoop(brain, reg)
        res = loop.run("Create note.txt with the content hello.")
        assert res.passed
        assert any(s.kind == "action" for s in res.steps)


class _FakeVLM:
    """Test double: a vision model too heavy to load in the unit suite."""

    def ask(self, image, question):
        from smallm.agent import VisionAnswer
        return VisionAnswer(question, "The image is a red circle.", 0.0)


def _write_png(path):
    from PIL import Image
    Image.new("RGB", (8, 8), "red").save(path)


def test_see_tool_grounds_vision_answer():
    with tempfile.TemporaryDirectory() as d:
        c = VirtualComputer(d)
        _write_png(os.path.join(d, "image.png"))
        from smallm.agent.tools import with_vision
        reg = with_vision(c, default_registry(c), _FakeVLM())
        brain = ScriptedBrain([
            {"thought": "look", "tool": "see",
             "args": {"path": "image.png", "question": "describe"}},
            {"thought": "ok", "answer": "The image shows a red circle."},
        ])
        loop = AgentLoop(brain, reg)
        res = loop.run("Look at image.png with the see tool.")
        assert res.passed
        assert "circle" in res.answer.lower()


def test_see_tool_reports_missing_image():
    with tempfile.TemporaryDirectory() as d:
        c = VirtualComputer(d)
        from smallm.agent.tools import with_vision
        reg = with_vision(c, default_registry(c), _FakeVLM())
        out = reg.call("see", {"path": "nope.png", "question": "describe"})
        assert '"ok": false' in out


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
