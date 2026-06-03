import numpy as np

from experiment import subsample, clamp_sizes, run_curve, format_experiment_report


def _fake_encode(texts):
    vecs = []
    for t in texts:
        low = t.lower()
        if "mail" in low:
            vecs.append([1.0, 0.0])
        elif "wiki" in low or "url" in low:
            vecs.append([0.0, 1.0])
        else:
            vecs.append([0.5, 0.5])
    return np.array(vecs, dtype=np.float32)


def _pool():
    return (
        [{"message": f"mail {i}", "agent": "gog"} for i in range(5)]
        + [{"message": f"url wiki {i}", "agent": "wikilm"} for i in range(5)]
    )


def test_subsample_is_stratified_and_deterministic():
    sub = subsample(_pool(), n=2, seed=1)
    counts = {}
    for r in sub:
        counts[r["agent"]] = counts.get(r["agent"], 0) + 1
    assert counts == {"gog": 2, "wikilm": 2}
    sub2 = subsample(_pool(), n=2, seed=1)
    assert [r["message"] for r in sub] == [r["message"] for r in sub2]


def test_clamp_sizes_caps_to_min_available():
    pool = (
        [{"message": f"mail {i}", "agent": "gog"} for i in range(3)]
        + [{"message": f"url {i}", "agent": "wikilm"} for i in range(6)]
    )
    clamped, cap = clamp_sizes([2, 4, 8], pool)
    assert cap == 3
    assert clamped == [2, 3]


def test_run_curve_one_entry_per_size_and_mechanism():
    pool = (
        [{"message": f"mail {i}", "agent": "gog"} for i in range(4)]
        + [{"message": f"url wiki {i}", "agent": "wikilm"} for i in range(4)]
    )
    test_set = [
        {"message": "mail test", "agent": "gog"},
        {"message": "url wiki test", "agent": "wikilm"},
        {"message": "totalement flou", "agent": "clarify"},
    ]
    results = run_curve(pool, test_set, sizes=[2, 3], threshold=0.55,
                        encode_fn=_fake_encode, seed=1)
    assert len(results) == 4
    mechs = {r["mechanism"] for r in results}
    assert mechs == {"prototype", "clf"}
    for r in results:
        assert 0.0 <= r["accuracy"] <= 1.0
        assert "clarify_recall" in r
        assert "per_agent" in r


def test_format_report_has_caveat_table_and_cost():
    results = [
        {"size": 3, "mechanism": "prototype", "accuracy": 0.8, "clarify_recall": 1.0},
        {"size": 3, "mechanism": "clf", "accuracy": 0.95, "clarify_recall": 1.0},
    ]
    report = format_experiment_report(results, "  modele: 10 in + 2 out tokens → 0.0001 $",
                                      min_accuracy=0.9, agent_names=["gog", "wikilm"], cap=3)
    assert "PLAFOND OPTIMISTE" in report or "plafond" in report.lower()
    assert "prototype" in report and "clf" in report
    assert "Coût" in report or "coût" in report.lower()
    assert "0.0001" in report
