"""Tests for the hand-over tooling: scripts/refresh_project_state.py and the two documents
docs/PROJECT_STATE.md and docs/HANDOFF_PROMPT.md.

The documents are what a NEW chat reads to continue the project, so they must stay truthful:
the generator must work, and the instructions must not point at files that do not exist.
"""
import importlib.util
import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATE = PROJECT_ROOT / "docs" / "PROJECT_STATE.md"
PROMPT = PROJECT_ROOT / "docs" / "HANDOFF_PROMPT.md"


def load_script():
    spec = importlib.util.spec_from_file_location("refresh_project_state", PROJECT_ROOT / "scripts" / "refresh_project_state.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rps = load_script()


def make_tree(root: Path, files: dict):
    for relative, content in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


# ---- list_files --------------------------------------------------------------------------------
def test_list_files_skips_secrets_caches_generated_output_and_placeholders(tmp_path):
    make_tree(tmp_path, {
        "README.md": "x", "backend/a.py": "x", "tests/test_a.py": "x", "data/raw/company.html": "x",
        ".env": "SECRET=1", ".venv/lib/x.py": "x", "backend/__pycache__/a.cpython-312.pyc": "x",
        ".pytest_cache/v/x": "x", "vector_db/chroma/db.sqlite3": "x", "data/processed/.gitkeep": "",
        "data/processed/chunks.json": "[]", "data/processed/parsed_document.json": "{}",
        "evaluation/questions.json": "[]", "evaluation/evaluation_report.md": "r", "evaluation/retrieval_results.json": "{}",
    })
    assert rps.list_files(tmp_path) == ["README.md", "backend/a.py", "data/raw/company.html",
                                        "evaluation/questions.json", "tests/test_a.py"]


def test_the_real_project_listing_never_contains_a_secret_or_cache():
    files = rps.list_files(PROJECT_ROOT)
    assert "backend/app.py" in files and "docs/PROJECT_STATE.md" in files
    assert not any(f == ".env" or "__pycache__" in f or f.startswith(".venv/") or f.startswith("vector_db/") for f in files)


# ---- describe ------------------------------------------------------------------------------------
def test_describe_returns_the_first_docstring_line(tmp_path):
    make_tree(tmp_path, {"a.py": '"""First line.\n\nMore text."""\nx = 1\n'})
    assert rps.describe(tmp_path, "a.py") == "First line."


def test_describe_is_safe_for_every_awkward_file(tmp_path):
    make_tree(tmp_path, {"none.py": "x = 1\n", "broken.py": "def (:\n", "readme.md": "# Title", "empty.py": ""})
    for name in ("none.py", "broken.py", "readme.md", "empty.py", "missing.py"):
        assert rps.describe(tmp_path, name) == ""


def test_describe_shortens_very_long_lines(tmp_path):
    make_tree(tmp_path, {"a.py": '"""' + "word " * 60 + '"""\n'})
    assert len(rps.describe(tmp_path, "a.py")) <= 100


# ---- render and replace ---------------------------------------------------------------------------------
def test_render_lists_files_and_totals_the_tests():
    text = rps.render_inventory(["a.py", "b.md"], {"a.py": "Does A."}, {"tests/test_x.py": 3, "tests/test_y.py": 4})
    assert "| `a.py` | Does A. |" in text and "| `b.md` |  |" in text
    assert "**Tests: 7 in total**" in text and "| `tests/test_x.py` | 3 |" in text and "| **Total** | **7** |" in text


def test_replace_block_changes_only_what_is_between_the_markers():
    original = f"intro\n\n{rps.START}\nOLD STUFF\n{rps.END}\n\noutro\n"
    result = rps.replace_block(original, "NEW STUFF")
    assert result == f"intro\n\n{rps.START}\nNEW STUFF\n{rps.END}\n\noutro\n"


@pytest.mark.parametrize("text", ["no markers at all", f"{rps.START} only the start", f"{rps.END}\n{rps.START}"])
def test_missing_or_reversed_markers_are_an_error(text):
    with pytest.raises(rps.StateFileError, match="markers"):
        rps.replace_block(text, "x")


# ---- refresh end to end (with a fake test counter, so pytest does not run itself) ------------------------------
def fake_counter(_root):
    return {"tests/test_a.py": 2}


def build_project(tmp_path):
    make_tree(tmp_path, {
        "backend/a.py": '"""Module A."""\n', "tests/test_a.py": '"""Tests."""\n',
        "docs/PROJECT_STATE.md": f"# State\n\nhand written\n\n{rps.START}\nstale\n{rps.END}\n\nmore hand written\n",
    })


def test_refresh_fills_the_block_and_leaves_the_rest_alone(tmp_path):
    build_project(tmp_path)
    assert rps.refresh(tmp_path, counter=fake_counter) is True
    text = (tmp_path / "docs" / "PROJECT_STATE.md").read_text(encoding="utf-8")
    assert "| `backend/a.py` | Module A. |" in text and "| **Total** | **2** |" in text
    assert text.startswith("# State\n\nhand written\n") and text.endswith("more hand written\n")
    assert "stale" not in text


def test_refresh_is_idempotent(tmp_path):
    build_project(tmp_path)
    rps.refresh(tmp_path, counter=fake_counter)
    assert rps.refresh(tmp_path, counter=fake_counter) is False


def test_check_mode_reports_without_writing(tmp_path):
    build_project(tmp_path)
    before = (tmp_path / "docs" / "PROJECT_STATE.md").read_text(encoding="utf-8")
    assert rps.refresh(tmp_path, write=False, counter=fake_counter) is True
    assert (tmp_path / "docs" / "PROJECT_STATE.md").read_text(encoding="utf-8") == before


def test_refresh_notices_a_new_file(tmp_path):
    build_project(tmp_path)
    rps.refresh(tmp_path, counter=fake_counter)
    make_tree(tmp_path, {"backend/new_module.py": '"""Brand new."""\n'})
    assert rps.refresh(tmp_path, write=False, counter=fake_counter) is True


def test_a_missing_state_file_is_a_clear_error(tmp_path):
    with pytest.raises(rps.StateFileError, match="does not exist"):
        rps.refresh(tmp_path, counter=fake_counter)


def test_the_real_test_counter_counts_per_file():
    """Runs the real `pytest --collect-only` on a tiny throw-away project."""
    import tempfile
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        make_tree(root, {"pytest.ini": "[pytest]\ntestpaths = tests\n",
                         "tests/test_one.py": "def test_a():\n    pass\n\ndef test_b():\n    pass\n",
                         "tests/test_two.py": "import pytest\n\n@pytest.mark.parametrize('n', [1, 2, 3])\ndef test_c(n):\n    pass\n"})
        assert rps.count_tests(root) == {"tests/test_one.py": 2, "tests/test_two.py": 3}


# ---- the real hand-over documents ----------------------------------------------------------------------------------
def test_state_file_has_the_markers_and_a_filled_inventory():
    text = STATE.read_text(encoding="utf-8")
    assert rps.START in text and rps.END in text
    block = text.split(rps.START, 1)[1].split(rps.END, 1)[0]
    assert "| `backend/rag/retriever.py` |" in block and "**Tests:" in block


@pytest.mark.parametrize("heading", [
    "## 3. Status", "## 4. Verified vs NOT verified", "## 6. Conventions", "## 11. Open questions",
    "## 13. Handoff protocol", "## 14. Traps already hit", "## 15. What to do NEXT", "## 16. How to finish a step",
])
def test_state_file_keeps_its_required_sections(heading):
    assert heading in STATE.read_text(encoding="utf-8")


def test_exactly_one_step_is_marked_next():
    rows = [line for line in STATE.read_text(encoding="utf-8").splitlines()
            if line.startswith("| ") and "**NEXT**" in line]
    assert len(rows) == 1


def test_documents_only_mention_scripts_that_exist_or_are_announced():
    announced = set()                                 # scripts planned for the NEXT step, mentioned as future work. UPDATE this at the end of every step.
    existing = {p.name for p in (PROJECT_ROOT / "scripts").glob("*.py")}
    for document in (STATE, PROMPT, PROJECT_ROOT / "README.md"):
        mentioned = set(re.findall(r"scripts/([a-z_]+\.py)", document.read_text(encoding="utf-8")))
        missing = mentioned - existing - announced
        assert not missing, f"{document.name} mentions scripts that do not exist: {sorted(missing)}"


def test_documents_only_mention_project_files_that_exist():
    state = STATE.read_text(encoding="utf-8")
    for path in set(re.findall(r"`((?:backend|tests|docs|data|evaluation)/[A-Za-z0-9_/\.]+\.(?:py|md|json|html))`", state)):
        if "*" in path or path.startswith("data/processed") or path in (
                "evaluation/retrieval_results.json", "evaluation/evaluation_report.md"):
            continue                                   # generated files do not have to exist
        assert (PROJECT_ROOT / path).exists() or path == "backend/rag/llm.py" or path.endswith("prompt_builder.py") \
            or path.endswith(("relevance.py", "answer_validator.py")), f"PROJECT_STATE.md mentions {path}, which does not exist"


def test_prompt_contains_the_whole_method_and_the_task_line():
    text = PROMPT.read_text(encoding="utf-8")
    assert text.count("=== COPY FROM HERE ===") == 1 and text.count("=== COPY UNTIL HERE ===") == 1
    for must in ("PROJECT_STATE.md", "ProjectDocumentation.docx", "mutation", "changes_only", "full_stepN",
                 "baseline", "fresh virtual environment", "Honesty", "Never paste", "TASK (I fill this in)"):
        assert must.lower() in text.lower(), f"the hand-off prompt lost: {must}"
