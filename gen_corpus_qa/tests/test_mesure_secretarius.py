from mesure_secretarius import confusion_matrix, taux_commandes_volees, rappel


def test_confusion_matrix_compte():
    pairs = [("wiki", "wiki"), ("wiki", "secretarius"),
             ("gog", "gog"), ("secretarius", "secretarius"),
             ("null", "wiki")]
    m = confusion_matrix(pairs)
    assert m["wiki"]["wiki"] == 1
    assert m["wiki"]["secretarius"] == 1
    assert m["gog"]["gog"] == 1
    assert m["null"]["wiki"] == 1


def test_taux_commandes_volees():
    # 1 wiki volé + 0 gog volé sur 2 commandes = 0.5
    pairs = [("wiki", "secretarius"), ("gog", "gog")]
    assert taux_commandes_volees(confusion_matrix(pairs)) == 0.5


def test_rappel():
    pairs = [("secretarius", "secretarius"), ("secretarius", "null")]
    assert rappel(confusion_matrix(pairs), "secretarius") == 0.5


def test_matrices_vides_ne_plantent_pas():
    m = confusion_matrix([])
    assert taux_commandes_volees(m) == 0.0
    assert rappel(m, "secretarius") == 0.0
