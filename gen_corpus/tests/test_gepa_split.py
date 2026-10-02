from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_split_disjoint_taille_et_deterministe():
    from promptGenGEPA import split_trainset
    ex = list(range(100))
    train, val = split_trainset(ex, 40, seed=0)
    assert len(val) == 40 and len(train) == 60 and set(train).isdisjoint(val)
    assert split_trainset(ex, 40, seed=0) == (train, val)


def test_split_borne_si_trop_peu_d_exemples():
    from promptGenGEPA import split_trainset
    train, val = split_trainset(list(range(10)), 40)
    assert len(val) == 5 and len(train) == 5
