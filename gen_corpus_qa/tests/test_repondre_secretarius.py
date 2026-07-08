import repondre_secretarius as rs


def test_met_le_scale_a_zero_et_injecte_le_document(monkeypatch):
    appels = {}

    def fake_set_scale(base_url, scale):
        appels["scale"] = scale
        appels["scale_url"] = base_url

    def fake_infer(base_url, document, question):
        appels["document"] = document
        appels["question"] = question
        appels["infer_url"] = base_url
        return "réponse simulée"

    monkeypatch.setattr(rs, "set_lora_scale", fake_set_scale)
    monkeypatch.setattr(rs, "infer_llama", fake_infer)

    out = rs.repondre_secretarius("quel modèle vous anime ?", base_url="http://x:8996")

    assert out == "réponse simulée"
    assert appels["scale"] == 0.0                 # phi-4 nu
    assert appels["scale_url"] == "http://x:8996"
    assert appels["infer_url"] == "http://x:8996"
    assert "sanroque" in appels["document"]       # document injecté
    assert appels["question"] == "quel modèle vous anime ?"
