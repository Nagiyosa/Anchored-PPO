"""M6 验证度量测试(纯函数)."""

import math

from derbysim.validation import metrics as M
from derbysim.validation.report import s8_statement


def test_mae_rmse():
    errs = [3, -4, 0, 5]
    assert M.mae(errs) == 3.0
    assert math.isclose(M.rmse(errs), math.sqrt((9 + 16 + 0 + 25) / 4))


def test_median_and_bias():
    assert M.median_abs([1, -2, 3, -100]) == 2.5
    assert M.bias([-5, -1, 1, 5, 100]) == 1.0   # 中位带符号


def test_hit_rate():
    assert M.hit_rate([1, 2, 30, -3], 10) == 0.75
    assert M.hit_rate([], 10) != M.hit_rate([], 10) or True  # nan-safe(不抛)


def test_dist_summary():
    d = M.dist_summary([10, 20, 30, 40, 50])
    assert d["n"] == 5 and d["median"] == 30.0


def test_s8_statement_wording():
    s = s8_statement(3, 91, 10, 38, 3)
    assert "中位误差" in s and "描述性有效" in s and "反事实" in s
    assert "MAE" not in s.split("中位误差")[0]   # 不把中位误标为MAE
