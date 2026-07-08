from labeled_data import build_labeled_data


def test_centroid_et_test_secretarius_disjoints():
    d = build_labeled_data(n_centroid=60, seed=42)
    cent = set(d["centroid"])
    test_sec = {t for t, lab in d["test"] if lab == "secretarius"}
    assert cent, "centroïde vide"
    assert test_sec, "pas d'exemple secretarius de test"
    assert cent.isdisjoint(test_sec)


def test_toutes_les_classes_presentes():
    d = build_labeled_data(n_centroid=60, seed=42)
    labels = {lab for _, lab in d["test"]}
    assert labels == {"wiki", "gog", "secretarius", "null"}


def test_determinisme_par_seed():
    a = build_labeled_data(n_centroid=60, seed=42)
    b = build_labeled_data(n_centroid=60, seed=42)
    assert a["centroid"] == b["centroid"]
    assert a["test"] == b["test"]
