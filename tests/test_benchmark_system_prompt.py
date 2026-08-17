from __future__ import annotations

from remtp.gsm8k_benchmark import benchmark_messages as gsm8k_messages
from remtp.humaneval_benchmark import benchmark_messages as humaneval_messages


def test_mimo_empty_system_prompt_is_explicit() -> None:
    expected = [
        {"role": "system", "content": ""},
        {"role": "user", "content": "hello"},
    ]
    assert gsm8k_messages("hello", empty_system_prompt=True) == expected
    assert humaneval_messages("hello", empty_system_prompt=True) == expected


def test_default_prompt_does_not_invent_a_system_message() -> None:
    expected = [{"role": "user", "content": "hello"}]
    assert gsm8k_messages("hello", empty_system_prompt=False) == expected
    assert humaneval_messages("hello", empty_system_prompt=False) == expected
