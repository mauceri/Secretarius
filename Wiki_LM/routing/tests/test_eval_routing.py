from router_base import Router, RouteResult
from eval_routing import stratified_split, evaluate


def _corpus():
    return [
        {"message": "m1", "agent": "gog"},
        {"message": "m2", "agent": "gog"},
        {"message": "m3", "agent": "gog"},
        {"message": "m4", "agent": "gog"},
        {"message": "w1", "agent": "wikilm"},
        {"message": "w2", "agent": "wikilm"},
        {"message": "w3", "agent": "wikilm"},
        {"message": "w4", "agent": "wikilm"},
    ]


def test_split_is_stratified_and_deterministic():
    train, test = stratified_split(_corpus(), test_frac=0.5, seed=1)
    test_agents = sorted(r["agent"] for r in test)
    assert test_agents == ["gog", "gog", "wikilm", "wikilm"]
    train2, test2 = stratified_split(_corpus(), test_frac=0.5, seed=1)
    assert [r["message"] for r in test] == [r["message"] for r in test2]
    assert set(r["message"] for r in train).isdisjoint(r["message"] for r in test)


class _AlwaysGog(Router):
    def route(self, message):
        return RouteResult("gog", 0.42)


def test_evaluate_metrics():
    test_set = [
        {"message": "a", "agent": "gog"},
        {"message": "b", "agent": "gog"},
        {"message": "c", "agent": "wikilm"},
        {"message": "d", "agent": "wikilm"},
    ]
    report = evaluate(_AlwaysGog(), test_set)
    assert report.accuracy == 0.5
    assert report.per_agent["gog"] == 1.0
    assert report.per_agent["wikilm"] == 0.0
    assert report.confusion[("gog", "gog")] == 2
    assert report.confusion[("wikilm", "gog")] == 2
    assert len(report.misroutes) == 2
    assert report.misroutes[0]["expected"] == "wikilm"
    assert report.misroutes[0]["predicted"] == "gog"
    assert report.misroutes[0]["confidence"] == 0.42


def test_evaluate_empty_test_set():
    report = evaluate(_AlwaysGog(), [])
    assert report.accuracy == 0.0
    assert report.misroutes == []
