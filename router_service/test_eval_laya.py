from router_service.eval_laya import choisir_seuil, metriques


def test_metriques():
    cl = [({"/q"}, [("/q", 0.9), ("/c", 0.1)]),            # décidé juste
          ({"/lire"}, [("/chercher", 0.5), ("/lire", 0.4)]),  # question pertinente
          ({"aucune"}, [("/q", 0.8), (None, 0.2)]),       # décidé faux
          ({"/c"}, [("/q", 0.5), ("/tags", 0.3)])]        # question non pertinente
    m = metriques(cl, 0.6)
    assert m["decide_juste"] == 0.5 and m["taux_question"] == 0.5 and m["juste_ou_question"] == 0.5
    assert m["par_commande"]["/q"] == 1.0 and m["par_commande"]["aucune"] == 0.0


def test_choisir_seuil():
    cl = [({"/q"}, [("/q", 0.9), ("/c", 0.1)])] * 18 + [({"/c"}, [("/q", 0.55), ("/c", 0.45)])] * 2
    assert choisir_seuil(cl) == 0.6


def test_attendu_validation():
    from router_service.eval_laya import attendu_validation
    assert attendu_validation({"answers": {"domaine": "wiki", "commande_wiki": "/q"}}) == {"/q"}
    assert attendu_validation({"answers": {"domaine": "planification"}}) == {"/cron"}
    assert attendu_validation({"answers": {"domaine": "aucune"}}) == {"aucune"}


def test_reponse_du_routeur_traduite_en_classement():
    from router_service.eval_laya import _route_vers_classement, metriques
    cl = [({"/q"}, _route_vers_classement({"status": "ok", "command": "/q", "args": "x"})),
          ({"aucune"}, _route_vers_classement({"status": "no_match"})),
          ({"/lire"}, _route_vers_classement({"status": "clarify", "args": "x",
                                              "options": [{"command": "/chercher"}, {"command": "/lire"}]}))]
    m = metriques(cl, 0.5)
    assert m["juste_ou_question"] == 1.0 and abs(m["taux_question"] - 1 / 3) < 1e-9


def test_question_a_trois_options_toutes_comptees():
    from router_service.eval_laya import _route_vers_classement, metriques
    cl = [({"aucune"}, _route_vers_classement({"status": "clarify", "args": "x", "options": [
        {"command": "/q"}, {"command": "/c"}, {"command": None}]}))]
    assert metriques(cl, 0.5)["juste_ou_question"] == 1.0
