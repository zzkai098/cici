"""Unit tests for the roadmap-1 tool surface.

These run without touching the API: tools are plain objects, and the async
contract is exercised through Registry.run_all the way the agent uses it.
"""

import asyncio

import pytest

from cici.tools import default_registry
from cici.tools.edit import EditTool
from cici.tools.read import ReadTool
from cici.tools.workspace import Workspace
from cici.tools.write import WriteTool


@pytest.fixture
def ws(tmp_path):
    return Workspace(tmp_path)


def run(coro):
    return asyncio.run(coro)


# --- Workspace.resolve: the confinement boundary ------------------------------


def test_resolve_accepts_paths_inside_the_root(ws):
    assert ws.resolve("a/b.py") == ws.root / "a" / "b.py"


def test_resolve_rejects_parent_traversal(ws):
    with pytest.raises(ValueError, match="escapes the workspace"):
        ws.resolve("../outside.py")


def test_resolve_rejects_absolute_paths_outside(ws):
    with pytest.raises(ValueError, match="escapes the workspace"):
        ws.resolve("/etc/passwd")


def test_resolve_rejects_sibling_with_shared_prefix(tmp_path):
    """cici_101 used startswith(), which accepts /home/me/app-secrets when the
    root is /home/me/app. is_relative_to compares components, so it must not."""
    root = tmp_path / "app"
    root.mkdir()
    (tmp_path / "app-secrets").mkdir()
    ws = Workspace(root)
    with pytest.raises(ValueError, match="escapes the workspace"):
        ws.resolve("../app-secrets/keys.txt")


def test_resolve_rejects_symlink_pointing_outside(tmp_path):
    root = tmp_path / "app"
    root.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("shh")
    (root / "link.txt").symlink_to(secret)
    with pytest.raises(ValueError, match="escapes the workspace"):
        Workspace(root).resolve("link.txt")


# --- read ---------------------------------------------------------------------


def test_read_numbers_lines(ws):
    (ws.root / "f.py").write_text("alpha\nbeta\n")
    out = run(ReadTool(ws).run(path="f.py"))
    assert out.splitlines() == ["1\talpha", "2\tbeta"]


def test_read_trailing_newline_is_not_a_line(ws):
    (ws.root / "f.py").write_text("only\n")
    assert run(ReadTool(ws).run(path="f.py")) == "1\tonly"


def test_read_offset_and_limit_report_the_remainder(ws):
    (ws.root / "f.py").write_text("\n".join(f"L{i}" for i in range(1, 11)) + "\n")
    out = run(ReadTool(ws).run(path="f.py", offset=3, limit=2))
    assert "3\tL3" in out and "4\tL4" in out
    assert "offset=5" in out and "of 10" in out


def test_read_missing_file(ws):
    with pytest.raises(FileNotFoundError):
        run(ReadTool(ws).run(path="nope.py"))


def test_read_directory_points_at_bash(ws):
    (ws.root / "sub").mkdir()
    with pytest.raises(IsADirectoryError, match="use bash"):
        run(ReadTool(ws).run(path="sub"))


# --- write --------------------------------------------------------------------


def test_write_creates_with_parents(ws):
    out = run(WriteTool(ws).run(path="deep/dir/f.py", content="x = 1\n"))
    assert (ws.root / "deep/dir/f.py").read_text() == "x = 1\n"
    assert out.startswith("created")


def test_write_overwrites_and_says_so(ws):
    (ws.root / "f.py").write_text("old\n")
    out = run(WriteTool(ws).run(path="f.py", content="new\n"))
    assert (ws.root / "f.py").read_text() == "new\n"
    assert out.startswith("overwrote")


def test_write_overwrite_is_undoable(ws):
    (ws.root / "f.py").write_text("original\n")
    run(WriteTool(ws).run(path="f.py", content="clobbered\n"))
    run(EditTool(ws).run(command="undo_edit", path="f.py"))
    assert (ws.root / "f.py").read_text() == "original\n"


# --- edit: str_replace --------------------------------------------------------


def test_str_replace_swaps_one_occurrence(ws):
    (ws.root / "f.py").write_text("a\nTARGET\nb\n")
    out = run(
        EditTool(ws).run(
            command="str_replace", path="f.py", old_str="TARGET", new_str="DONE"
        )
    )
    assert (ws.root / "f.py").read_text() == "a\nDONE\nb\n"
    assert "line 2" in out


def test_str_replace_refuses_ambiguous_match(ws):
    """The reason this check exists: replacing a string that appears twice
    corrupts the file in a way the model cannot see."""
    (ws.root / "f.py").write_text("dup\ndup\n")
    with pytest.raises(ValueError, match="appears 2 times"):
        run(
            EditTool(ws).run(
                command="str_replace", path="f.py", old_str="dup", new_str="x"
            )
        )
    assert (ws.root / "f.py").read_text() == "dup\ndup\n"  # untouched


def test_str_replace_refuses_missing_match(ws):
    (ws.root / "f.py").write_text("a\n")
    with pytest.raises(ValueError, match="not found"):
        run(
            EditTool(ws).run(
                command="str_replace", path="f.py", old_str="zzz", new_str="x"
            )
        )


def test_str_replace_empty_new_str_deletes(ws):
    (ws.root / "f.py").write_text("keep\ndrop\n")
    run(
        EditTool(ws).run(
            command="str_replace", path="f.py", old_str="drop\n", new_str=""
        )
    )
    assert (ws.root / "f.py").read_text() == "keep\n"


# --- edit: insert -------------------------------------------------------------


def test_insert_after_line(ws):
    (ws.root / "f.py").write_text("one\nthree\n")
    run(EditTool(ws).run(command="insert", path="f.py", insert_line=1, new_str="two"))
    assert (ws.root / "f.py").read_text() == "one\ntwo\nthree\n"


def test_insert_at_top(ws):
    (ws.root / "f.py").write_text("body\n")
    run(
        EditTool(ws).run(command="insert", path="f.py", insert_line=0, new_str="header")
    )
    assert (ws.root / "f.py").read_text() == "header\nbody\n"


def test_insert_at_end_of_file_without_trailing_newline(ws):
    (ws.root / "f.py").write_text("last")
    run(EditTool(ws).run(command="insert", path="f.py", insert_line=1, new_str="added"))
    assert (ws.root / "f.py").read_text() == "last\nadded\n"


def test_insert_out_of_range(ws):
    (ws.root / "f.py").write_text("one\n")
    with pytest.raises(IndexError, match="out of range"):
        run(EditTool(ws).run(command="insert", path="f.py", insert_line=9, new_str="x"))


# --- edit: undo ---------------------------------------------------------------


def test_undo_walks_back_one_edit_at_a_time(ws):
    f = ws.root / "f.py"
    f.write_text("v1\n")
    edit = EditTool(ws)
    run(edit.run(command="str_replace", path="f.py", old_str="v1", new_str="v2"))
    run(edit.run(command="str_replace", path="f.py", old_str="v2", new_str="v3"))
    assert f.read_text() == "v3\n"
    run(edit.run(command="undo_edit", path="f.py"))
    assert f.read_text() == "v2\n"
    run(edit.run(command="undo_edit", path="f.py"))
    assert f.read_text() == "v1\n"


def test_undo_with_nothing_to_undo(ws):
    (ws.root / "f.py").write_text("x\n")
    with pytest.raises(FileNotFoundError, match="no edit to undo"):
        run(EditTool(ws).run(command="undo_edit", path="f.py"))


def test_backups_are_not_created_until_a_file_is_edited(ws):
    assert not ws.backup_dir.exists()
    ReadTool(ws)
    EditTool(ws)
    assert not ws.backup_dir.exists(), "cici_101 made .backups/ on construction"


# --- bash ---------------------------------------------------------------------


def test_bash_runs_and_reports_exit_code(ws, monkeypatch):
    monkeypatch.setenv("CICI_YOLO", "1")
    out = run(default_registry(ws).dispatch("bash", {"command": "echo hi"}))
    assert "exit code: 0" in out and "hi" in out


def test_bash_captures_failure(ws, monkeypatch):
    monkeypatch.setenv("CICI_YOLO", "1")
    out = run(default_registry(ws).dispatch("bash", {"command": "exit 3"}))
    assert "exit code: 3" in out


def test_bash_runs_in_the_workspace_root(ws, monkeypatch):
    monkeypatch.setenv("CICI_YOLO", "1")
    (ws.root / "marker.txt").write_text("")
    out = run(default_registry(ws).dispatch("bash", {"command": "ls"}))
    assert "marker.txt" in out


def test_bash_timeout_kills_the_command(ws, monkeypatch):
    monkeypatch.setenv("CICI_YOLO", "1")
    with pytest.raises(TimeoutError, match="exceeded 1s"):
        run(
            default_registry(ws).dispatch("bash", {"command": "sleep 30", "timeout": 1})
        )


def test_bash_refuses_without_a_tty_and_without_yolo(ws, monkeypatch):
    monkeypatch.delenv("CICI_YOLO", raising=False)
    out = run(default_registry(ws).dispatch("bash", {"command": "echo hi"}))
    assert "refused" in out and "CICI_YOLO" in out


# --- the contract the agent actually uses -------------------------------------


def test_run_all_contains_a_failing_tool(ws):
    """A tool that raises must come back as an is_error tool_result, not a crash."""
    import types

    reg = default_registry(ws)
    msg = types.SimpleNamespace(
        content=[
            types.SimpleNamespace(
                type="tool_use", id="t1", name="read", input={"path": "missing.py"}
            ),
            types.SimpleNamespace(
                type="tool_use",
                id="t2",
                name="write",
                input={"path": "ok.py", "content": "fine\n"},
            ),
        ]
    )
    blocks = run(reg.run_all(msg))
    assert [b["is_error"] for b in blocks] == [True, False]
    assert (ws.root / "ok.py").exists()


def test_every_registered_tool_has_described_parameters(ws):
    """Hand-written schemas are a convention in this repo; keep them honest."""
    for schema in default_registry(ws).schemas():
        assert schema["description"], schema["name"]
        for param, spec in schema["input_schema"]["properties"].items():
            assert spec.get("description"), f"{schema['name']}.{param}"
