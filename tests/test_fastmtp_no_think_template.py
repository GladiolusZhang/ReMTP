from pathlib import Path

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]


def _render(enable_thinking: bool) -> str:
    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / "models" / "FastMTP",
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer.chat_template = (
        ROOT / "configs" / "fastmtp_no_think_chat_template.jinja"
    ).read_text(encoding="utf-8")
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": ""},
            {"role": "user", "content": "Answer directly."},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def test_fastmtp_no_think_template_prefills_closed_think_block() -> None:
    prompt = _render(False)
    assert prompt.endswith(
        "<|im_start|>assistant\n<think>\n\n</think>\n"
    )


def test_fastmtp_thinking_template_opens_think_block() -> None:
    prompt = _render(True)
    assert prompt.endswith("<|im_start|>assistant\n<think>\n")
