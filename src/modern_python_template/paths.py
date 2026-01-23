"""Provides a centralized way to get the root directory of the project.

The days of using garbage like `Path(__file__).parent.parent ` and `os.getcwd()` are over.
Use fixed, immutable paths throughout the codebase.
Your scripts behaviour should never change based on the current working directory.

Since you know there will always be a uv.lock file in the root of the project, we always use that as an anchor.

Notes:
- You may consider using `pyproject.toml` as your anchor, but if you begin using `uv` workspaces, you will have
    multiple sub-directories with `pyproject.toml` files.
- This may seem awkward since to define paths to data or output files, you will need to use something like:
        ```python
        DATA_INPUT_DIR = PROJECT_ROOT / "src" / "modern_python_template" / "data" / "input" / "my_data.csv"
        ```
    This is intentional, you probably shouldn't be have data or output files in the src directory.
    Instead, you should have a data directory outside of your src layout:
    ```python
    DATA_INPUT_DIR = PROJECT_ROOT / "data" / "input" / "my_data.csv"
    ```
    This way, any refactoring you do of the code will not break the paths to the data files.

"""

from pathlib import Path
from typing import Final

from pyprojroot import find_root, has_file

PROJECT_ROOT: Final[Path] = find_root(has_file("uv.lock"))
