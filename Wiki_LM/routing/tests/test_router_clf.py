import numpy as np

from router_clf import ClfRouter


def _fake_encode(texts):
    """Encodeur 2-D déterministe : gog→[1,0], wikilm→[0,1], autre→[0.5,0.5]."""
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


def _train():
    return [
        {"message": "envoie un mail", "agent": "gog"},
        {"message": "lis mon mail", "agent": "gog"},
        {"message": "mail urgent", "agent": "gog"},
        {"message": "capture cette url", "agent": "wikilm"},
        {"message": "ajoute au wiki", "agent": "wikilm"},
        {"message": "url wiki à garder", "agent": "wikilm"},
        {"message": "bla bla flou", "agent": "clarify"},
    ]


def test_excludes_clarify_from_classes():
    router = ClfRouter.from_corpus(_train(), threshold=0.55, encode_fn=_fake_encode)
    assert set(router.clf.classes_) == {"gog", "wikilm"}


def test_routes_clear_message():
    router = ClfRouter.from_corpus(_train(), threshold=0.55, encode_fn=_fake_encode)
    res = router.route("envoie un nouveau mail")
    assert res.agent == "gog"
    assert res.confidence > 0.55


def test_ambiguous_below_threshold_is_clarify():
    router = ClfRouter.from_corpus(_train(), threshold=0.55, encode_fn=_fake_encode)
    res = router.route("quelque chose de totalement flou")
    assert res.agent == "clarify"
