import subprocess
import unittest

from remtp.humaneval_evaluator import (
    build_candidate_source,
    docker_available,
    evaluate_source_docker,
)


TASK = {
    "task_id": "HumanEval/0",
    "prompt": "from typing import List\n\ndef add(a: int, b: int) -> int:\n    \"\"\"Add two integers.\"\"\"\n",
    "canonical_solution": "    return a + b\n",
    "test": "def check(candidate):\n    assert candidate(2, 3) == 5\n    assert candidate(-1, 1) == 0\n",
    "entry_point": "add",
}


def _image_available() -> bool:
    if not docker_available():
        return False
    result = subprocess.run(
        ["docker", "image", "inspect", "python:3-slim"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


class HumanEvalEvaluatorTest(unittest.TestCase):
    def test_builds_completion_source(self) -> None:
        record = {
            "candidate_mode": "completion",
            "candidate": "    return a + b\n",
        }
        source = build_candidate_source(TASK, record)
        self.assertIn("def add", source)
        self.assertIn("check(add)", source)
        compile(source, "<trusted-test>", "exec")

    def test_builds_full_function_without_duplicate_signature(self) -> None:
        record = {
            "candidate_mode": "full_function",
            "candidate": "def add(a, b):\n    return a + b\n",
        }
        source = build_candidate_source(TASK, record)
        self.assertEqual(source.count("def add"), 1)
        self.assertIn("from typing import List", source)

    @unittest.skipUnless(_image_available(), "Docker image is not installed")
    def test_docker_sandbox_distinguishes_pass_and_failure(self) -> None:
        good = build_candidate_source(
            TASK,
            {
                "candidate_mode": "completion",
                "candidate": "    return a + b\n",
            },
        )
        bad = build_candidate_source(
            TASK,
            {
                "candidate_mode": "completion",
                "candidate": "    return a - b\n",
            },
        )
        good_result = evaluate_source_docker(
            good,
            image="python:3-slim",
            timeout=5.0,
            memory="256m",
            cpus="1.0",
        )
        bad_result = evaluate_source_docker(
            bad,
            image="python:3-slim",
            timeout=5.0,
            memory="256m",
            cpus="1.0",
        )
        self.assertEqual(good_result["status"], "passed")
        self.assertEqual(bad_result["status"], "failed")


if __name__ == "__main__":
    unittest.main()
