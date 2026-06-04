import numpy as np

from router_embed import EmbedRouter


def _fake_encode(texts):
    """Encodeur 2-D déterministe basé sur des mots-clés."""
    vecs = []
    for t in texts:
        low = t.lower()
        if "mail" in low or "gog" in low:
            vecs.append([1.0, 0.0])
        elif "wiki" in low or "url" in low:
            vecs.append([0.0, 1.0])
        else:
            # ambigu : 45°, équidistant des deux axes — représente clarify
            vecs.append([0.7071, 0.7071])
    return np.array(vecs, dtype=np.float32)


def _train():
    return [
        {"message": "envoie un mail", "agent": "gog"},
        {"message": "cherche mon mail", "agent": "gog"},
        {"message": "capture cette url", "agent": "wikilm"},
        {"message": "ajoute au wiki", "agent": "wikilm"},
        {"message": "n'importe quoi", "agent": "clarify"},
        {"message": "aide-moi", "agent": "clarify"},
    ]


def test_includes_clarify_by_default():
    """clarify doit maintenant avoir son propre prototype (classe normale)."""
    router = EmbedRouter.from_corpus(_train(), encode_fn=_fake_encode)
    assert set(router.prototypes.keys()) == {"gog", "wikilm", "clarify"}


def test_routes_clear_message():
    router = EmbedRouter.from_corpus(_train(), encode_fn=_fake_encode)
    res = router.route("envoie un mail à Paul")
    assert res.agent == "gog"
    assert res.confidence > 0.9


def test_ambiguous_routes_to_clarify_prototype():
    """Un message ambigu doit être routé vers clarify via le prototype, pas le seuil."""
    router = EmbedRouter.from_corpus(_train(), encode_fn=_fake_encode)
    res = router.route("quelque chose de flou")
    assert res.agent == "clarify"


def test_exclude_clarify_still_works_for_legacy():
    """Passage explicite de exclude=('clarify',) : comportement hérité préservé."""
    router = EmbedRouter.from_corpus(_train(), threshold=0.8,
                                     encode_fn=_fake_encode, exclude=("clarify",))
    assert set(router.prototypes.keys()) == {"gog", "wikilm"}
    # Seuil de repli actif car clarify pas dans les prototypes
    res = router.route("quelque chose de flou")
    assert res.agent == "clarify"


def test_no_prototypes_is_clarify():
    router = EmbedRouter({}, threshold=0.5, encode_fn=_fake_encode)
    assert router.route("peu importe").agent == "clarify"
