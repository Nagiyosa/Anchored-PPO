"""M0 冒烟 + 区域常量测试."""

from derbysim import config


def test_import_and_version():
    import derbysim

    assert derbysim.__version__


def test_in_scope_size():
    assert len(config.IN_SCOPE_SIGNALS) == 48


def test_decision_points_size_and_subset():
    assert len(config.DECISION_POINTS) == 35
    assert config.DECISION_POINTS <= config.IN_SCOPE_SIGNALS


def test_decision_point_areas_partition():
    # 三区分组无重叠, 合并==35
    flat = [s for g in config.DECISION_POINTS_BY_AREA.values() for s in g]
    assert len(flat) == len(set(flat)) == 35


def test_static_data_paths_exist():
    for p in (
        config.DATA_DERBY_INFO,
        config.DATA_ROUTE_TC,
        config.DATA_PLATFORM_TC,
        config.DATA_PLATFORM_END_SIGNALS,
    ):
        assert p.exists(), f"缺静态数据文件: {p}"
