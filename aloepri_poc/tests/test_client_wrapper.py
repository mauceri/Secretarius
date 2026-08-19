import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from client_wrapper import ClientCodec


class FakeTokenizer:
    """Tokenizer jouet : un caractère = un token, pour tester sans réseau."""

    def encode(self, text):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def test_codec_round_trip_without_obfuscation_keys():
    permutation = {i: (i + 1) % 256 for i in range(256)}
    unpermute = {v: k for k, v in permutation.items()}
    codec = ClientCodec(permutation, unpermute, FakeTokenizer())

    original = "bonjour"
    encoded = codec.encode(original)
    assert encoded != [ord(c) for c in original]  # bien permuté, pas identité
    decoded = codec.decode(encoded)
    assert decoded == original


def test_codec_uses_the_permutation_in_the_same_direction_as_the_model():
    """Le sens compte : `encode` doit appliquer Π (clair → permuté) et `decode`
    son inverse. Un codec qui appliquerait la même table dans les deux sens
    passerait quand même le test de round-trip ci-dessus si Π était une
    involution — d'où une permutation volontairement non-involutive (+1 mod N)
    et la vérification des IDs eux-mêmes, pas seulement du texte final."""
    permutation = {i: (i + 1) % 256 for i in range(256)}
    unpermute = {v: k for k, v in permutation.items()}
    codec = ClientCodec(permutation, unpermute, FakeTokenizer())

    assert codec.encode("ab") == [ord("a") + 1, ord("b") + 1]
    assert codec.decode([ord("a") + 1, ord("b") + 1]) == "ab"
