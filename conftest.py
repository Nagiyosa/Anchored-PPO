"""让 `pytest` 与 `python -m pytest` 两种调用方式都能 import `derbysim`.

裸 `pytest`(或 `pytest tests/xxx.py`) **不会**把项目根放进 `sys.path` —— 只有 `python -m pytest`
会(它把 CWD 插到 sys.path[0])。根级 conftest.py 在收集前就被加载, 在这里显式把项目根前置,
两种方式行为一致, 也不依赖是否 `pip install -e .`。
"""
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
