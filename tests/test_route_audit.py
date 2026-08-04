from types import SimpleNamespace

import pandas as pd

from derbysim.learn.route_audit import (audit_operating_day_coverage,
                                         audit_training_route_chains,
                                         bc_dataset_fingerprint)


def _net():
    routes = {
        "r0": SimpleNamespace(route_id="r0", signal=1, end_signal=2, tc_sequence=["A", "B"]),
        "r1": SimpleNamespace(route_id="r1", signal=2, end_signal=3, tc_sequence=["B", "C"]),
        "loop": SimpleNamespace(route_id="loop", signal=3, end_signal=2, tc_sequence=["C", "A"]),
    }
    return SimpleNamespace(routes=routes, platforms={}, adjacency={1: {2}, 2: {3}, 3: {2}})


def _train(train_id, chain):
    return SimpleNamespace(train_id=train_id, route_chain=chain, stop_pids=set(), service_class="2")


def test_shared_track_circuit_is_not_mislabelled_as_a_loop():
    report = audit_training_route_chains([_train("shared", ["r0", "r1"]),
                                          _train("loop", ["r0", "r1", "r0"])],
                                         _net(), {})
    assert report["n_trains"] == 2
    assert report["route_count_including_entry"]["max"] == 3
    assert report["agent_route_count_excluding_entry"]["max"] == 2
    assert report["chains_with_repeated_body_tc_descriptive_only"] == 1
    assert report["chains_with_repeated_route_id"] == 1
    assert report["repeated_route_id_examples"][0]["train_id"] == "loop"
    assert report["chains_with_repeated_start_signal"] == 1
    budget = report["budget_candidates_including_entry"]
    assert budget["recommended_total_actual_route_cap"] == budget["three_times_train_p99"]


def test_operating_day_coverage_exposes_partial_boundaries():
    days = [{"start": pd.Timestamp("2023-09-01 03:00"),
             "end": pd.Timestamp("2023-09-02 03:00")},
            {"start": pd.Timestamp("2023-09-02 03:00"),
             "end": pd.Timestamp("2023-09-03 03:00")}]
    report = audit_operating_day_coverage(
        days, [17, 200], pd.Timestamp("2023-09-01 14:51"),
        pd.Timestamp("2023-09-03 03:00"))
    assert report["n_partial_boundary_days"] == 1
    assert report["partial_boundary_days"][0]["n_trains"] == 17


def test_bc_fingerprint_is_stable_to_row_order_and_changes_with_mask():
    rows = [
        {"split": "train", "date": "2023-09-01", "time": 1, "train_id": "A",
         "signal": 1, "chosen_route": "r0", "legal_route_ids": ["r0", "r1"], "obs": "{}"},
        {"split": "holdout", "date": "2023-11-01", "time": 2, "train_id": "B",
         "signal": 2, "chosen_route": "r1", "legal_route_ids": ["r1"], "obs": "{\"x\":1}"},
    ]
    a = bc_dataset_fingerprint(pd.DataFrame(rows))
    b = bc_dataset_fingerprint(pd.DataFrame(list(reversed(rows))))
    assert a == b
    rows[0]["legal_route_ids"] = ["r0"]
    c = bc_dataset_fingerprint(pd.DataFrame(rows))
    assert c["sha256_labels"] == a["sha256_labels"]
    assert c["sha256_legal_masks"] != a["sha256_legal_masks"]
    assert c["sha256_combined"] != a["sha256_combined"]
