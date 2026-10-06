# router_service/test_laya_classifier.py
from router_service.laya_classifier import classement


def _ans(dom, wiki=None, google=None):
    a = {"domaine": {"probabilities": dom}}
    if wiki:
        a["commande_wiki"] = {"probabilities": wiki}
    if google:
        a["commande_google"] = {"probabilities": google}
    return a


def test_classement_joint_trie():
    r = classement(_ans({"wiki": 0.7, "google": 0.2, "web": 0.05, "planification": 0.03, "aucune": 0.02},
                        wiki={"/q": 0.9, "/c": 0.1}, google={"/lire": 0.6, "/chercher": 0.4}))
    assert r[0] == ("/q", 0.63) and r[1] == ("/lire", 0.12)
    assert ("/c", 0.07) in r and ("/source", 0.05) in r and ("/cron", 0.03) in r and (None, 0.02) in r
    assert [p for _, p in r] == sorted((p for _, p in r), reverse=True)


def test_classement_sans_probabilites_de_niveau_2():
    r = classement(_ans({"wiki": 0.1, "google": 0.0, "web": 0.0, "planification": 0.9, "aucune": 0.0}))
    assert r[0] == ("/cron", 0.9)
