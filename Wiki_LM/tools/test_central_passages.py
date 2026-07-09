import numpy as np
from central_passages import pacsum_scores, select_central_passages, split_sentences


def test_split_sentences_fr():
    s = split_sentences("Bonjour le monde. Ceci est un test.")
    assert s == ["Bonjour le monde.", "Ceci est un test."]


def test_pacsum_favorise_la_phrase_centrale():
    # phrase 0 proche de 1 et 2 ; 1 et 2 éloignées entre elles -> 0 centrale
    e = np.array([[1.0, 0.0], [0.9, 0.436], [0.9, -0.436]], dtype=np.float32)
    # normaliser
    e = e / np.linalg.norm(e, axis=1, keepdims=True)
    sc = pacsum_scores(e, lambda1=1.0, lambda2=1.0, beta=0.0)
    assert sc.argmax() == 0


def test_texte_court_renvoye_tel_quel():
    txt = "Une note courte."
    assert select_central_passages(txt, budget_chars=6000) == "Une note courte."


def test_selection_respecte_budget_et_ordre():
    phrases = [f"Phrase numero {i} avec du texte de remplissage." for i in range(20)]
    texte = " ".join(phrases)

    def stub_embed(sents):
        # phrase i : vecteur one-hot bruité, la phrase 0 similaire à toutes (centrale)
        n = len(sents)
        m = np.eye(n, dtype=np.float32)
        m[:, 0] += 0.5  # tout le monde ressemble un peu à la phrase 0
        return m / np.linalg.norm(m, axis=1, keepdims=True)

    out = select_central_passages(texte, budget_chars=200, embed_fn=stub_embed)
    assert len(out) <= 200
    # ordre d'origine préservé : les numéros retenus sont croissants
    import re
    nums = [int(x) for x in re.findall(r"numero (\d+)", out)]
    assert nums == sorted(nums)
