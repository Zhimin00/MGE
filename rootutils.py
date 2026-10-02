"""Small local fallback for projects that expect the external rootutils package."""

import sys
from pathlib import Path


def setup_root(search_from, indicator=".project-root", pythonpath=True, cwd=False):
    path = Path(search_from).resolve()
    if path.is_file():
        path = path.parent

    for candidate in (path, *path.parents):
        if (candidate / indicator).exists():
            root = candidate
            break
    else:
        root = path

    if pythonpath and str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if cwd:
        import os

        os.chdir(root)
    return root
