import pytest
from labeled_data import build_labeled_data
from classify_secretarius import SecretariusClassifier


@pytest.fixture(scope="module")
def clf():
    data = build_labeled_data(n_centroid=60, seed=42)
    return SecretariusClassifier(data["centroid"])


def test_sortie_toujours_dans_les_quatre_classes(clf):
    for msg in ["ingère le wiki", "cherche les mails de Paul",
                "quel modèle vous anime ?", "il fait beau aujourd'hui"]:
        assert clf.classify(msg) in {"wiki", "gog", "secretarius", "null"}


def test_commande_gog_evidente_non_volee_par_secretarius(clf):
    # propriété critique : la règle de priorité protège le routage existant.
    assert clf.classify("cherche les mails de Paul cette semaine") != "secretarius"


def test_seuil_haut_desactive_secretarius():
    # règle déterministe : avec un seuil > 1 (cosinus max de vecteurs
    # normalisés = 1), la condition sim_sec >= seuil est toujours fausse,
    # donc secretarius ne gagne jamais — teste le code, pas la sémantique.
    data = build_labeled_data(n_centroid=60, seed=42)
    clf_strict = SecretariusClassifier(data["centroid"], seuil=1.01)
    for txt, _ in data["test"][:30]:
        assert clf_strict.classify(txt) != "secretarius"
