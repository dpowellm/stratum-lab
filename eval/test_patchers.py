"""Eval script for patcher unit tests.

Runs pytest on tests/test_patchers.py via subprocess and captures output
to eval/outputs/patcher-unit-tests.txt.
"""

import subprocess
from pathlib import Path


def main():
    output_path = Path(__file__).parent / "outputs" / "patcher-unit-tests.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        ["python", "-m", "pytest", "tests/test_patchers.py", "-v", "--tb=short"],
        capture_output=True, text=True, cwd=str(Path(__file__).parent.parent)
    )

    output = result.stdout + "\n" + result.stderr
    output_path.write_text(output)
    print(output)


if __name__ == "__main__":
    main()
