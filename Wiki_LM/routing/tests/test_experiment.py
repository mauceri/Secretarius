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


def test_run_curve_one_entry_per_size():
    pool = (
        [{"message": f"mail {i}", "agent": "gog"} for i in range(4)]
        + [{"message": f"url wiki {i}", "agent": "wikilm"} for i in range(4)]
    )
    test_set = [
        {"message": "mail test", "agent": "gog"},
        {"message": "url wiki test", "agent": "wikilm"},
    ]
    results = run_curve(pool, test_set, sizes=[2, 3], threshold=0.55,
                        encode_fn=_fake_encode, seed=1)
    # Un seul routeur (prototype) → 1 entrée par taille
    assert len(results) == 2
    for r in results:
        assert 0.0 <= r["accuracy"] <= 1.0
        assert "clarify_recall" in r
        assert "per_agent" in r


def test_format_report_has_caveat_table_and_cost():
    results = [
        {"size": 3, "accuracy": 0.8, "per_agent": {"gog": 1.0}, "clarify_recall": 1.0},
        {"size": 6, "accuracy": 0.95, "per_agent": {"gog": 1.0}, "clarify_recall": 1.0},
    ]
    report = format_experiment_report(results, "  modele: 10 in + 2 out tokens → 0.0001 $",
                                      min_accuracy=0.9, agent_names=["gog", "wikilm"], cap=6)
    assert "PLAFOND OPTIMISTE" in report or "plafond" in report.lower()
    assert "94.7" not in report  # pas de mécanisme nommé
    assert "Coût" in report or "coût" in report.lower()
    assert "0.0001" in report


def test_format_report_with_diagnostics():
    results = [{"size": 3, "accuracy": 0.9, "per_agent": {}, "clarify_recall": 1.0}]
    diag = {
        "min_correct_sim": 0.72,
        "inter_proto": [[1.0, 0.3], [0.3, 1.0]],
        "proto_names": ["gog", "wikilm"],
        "prior_sim": [[1.0, 0.25], [0.25, 1.0]],
        "desc_names": ["gog", "wikilm"],
    }
    report = format_experiment_report(results, "coût: 0 $", min_accuracy=0.9,
                                      agent_names=["gog", "wikilm"], cap=3,
                                      diagnostics=diag)
    assert "0.720" in report
    assert "inter-prototype" in report.lower() or "inter-proto" in report.lower()
    assert "a priori" in report.lower() or "priori" in report.lower()
    assert "0.30" in report  # similarité inter-classe
