"""MBPP prompt, sampling, code extraction and executed module (2026-11-16)."""
from __future__ import annotations

import json

from remtp import mbpp_benchmark as mbpp

ROW = {"task_id": 11, "text": "Write a python function to remove first and last occurrence of a given character.",
       "code": "", "test_setup_code": "import re",
       "test_list": ['assert remove_Occ("hello","l") == "heo"', 'assert remove_Occ("abcda","a") == "bcd"',
                     'assert remove_Occ("PHP","P") == "H"']}


def test_prompt_lists_the_three_assertions_and_the_task():
    prompt = mbpp.build_prompt(ROW)
    for test in ROW["test_list"]:
        assert test in prompt
    assert prompt.endswith("Task: " + ROW["text"])


def test_sampling_uses_the_test_split_deterministically(tmp_path):
    rows = [dict(ROW, task_id=i) for i in range(1, 975)]
    path = tmp_path / "mbpp.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    loaded = mbpp.load_mbpp(path)
    a = mbpp.sample_mbpp(loaded, 50, 7)
    b = mbpp.sample_mbpp(loaded, 50, 7)
    assert [r["task_id"] for r in a] == [r["task_id"] for r in b]
    assert all(11 <= r["task_id"] <= 510 for r in a)
    assert [r["task_id"] for r in a] == sorted(r["task_id"] for r in a)


def test_code_extraction_and_source():
    reply = "Here you go:\n```python\ndef remove_Occ(s, ch):\n    return s\n```\nDone."
    code = mbpp.extract_code(reply)
    assert code == "def remove_Occ(s, ch):\n    return s"
    assert mbpp.extract_code("<think>x</think>def f():\n    pass") == "def f():\n    pass"
    source = mbpp.build_source(ROW, code)
    assert source.startswith("import re\n\ndef remove_Occ")
    assert source.rstrip().endswith('assert remove_Occ("PHP","P") == "H"')


def test_executed_module_passes_and_fails_as_expected():
    good = "def remove_Occ(s, ch):\n    s = s.replace(ch, '', 1)\n    i = s.rfind(ch)\n    return s if i < 0 else s[:i] + s[i+1:]"
    namespace: dict = {}
    exec(compile(mbpp.build_source(ROW, good), "<t>", "exec"), namespace)
    bad = "def remove_Occ(s, ch):\n    return s"
    try:
        exec(compile(mbpp.build_source(ROW, bad), "<t>", "exec"), {})
    except AssertionError:
        pass
    else:
        raise AssertionError("wrong solution passed")
