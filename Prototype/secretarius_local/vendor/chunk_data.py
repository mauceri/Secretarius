#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chunker sémantique + nettoyage + filtrage (JSONL -> JSONL)

Entrée (JSONL) : chaque ligne = {"source":..., "titre":..., "contenu":..., ...}
Sortie (JSONL) : chaque ligne = {"source":..., "titre":..., "chunks":[str,...], ...}

Objectifs :
- Nettoyer des artefacts fréquents (préfixes b"…", b'…', guillemets, espaces, etc.)
- Filtrer/fusionner les micro-chunks (trop courts / trop pauvres)
- Chunking réellement sémantique via embeddings de phrases (SentenceTransformer)

Dépendances :
- nltk
- sentence-transformers
- numpy
- tqdm
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

import nltk
from sentence_transformers import SentenceTransformer

# ROCm iGPU (ex. Ryzen 680M) a parfois besoin de l'override pour matcher l'arch.
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")


# ----------------------------
# NLTK
# ----------------------------

def ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")


# ----------------------------
# Nettoyage
# ----------------------------

_BYTES_PREFIX_RE = re.compile(r"^\s*b([\"'])", re.DOTALL)  # b".... or b'....
_LEADING_BQ_RE = re.compile(r"^[\"']{1,3}\s*")            # quotes
_TRAILING_BQ_RE = re.compile(r"\s*[\"']{1,3}$")           # quotes
_MULTI_NL_RE = re.compile(r"\n{3,}")
_MULTI_SP_RE = re.compile(r"[ \t]{2,}")
# Proportion de "bruit" (caractères non alphanumériques, hors ponctuation légère)
_NOISE_CHARS_RE = re.compile(r"[^0-9A-Za-zÀ-ÖØ-öø-ÿ \n\.\,\;\:\!\?\(\)\[\]\-–—'’\"%°/]+")


import codecs

def clean_text(raw: str) -> str:
    if raw is None:
        return ""

    s = str(raw)

    # 1) Suppression préfixe bytes b" / b'
    s = _BYTES_PREFIX_RE.sub(r"\1", s)

    # 1bis) Déséchappement si la chaîne ressemble à une "repr" (\" et \\n littéraux)
    # On limite pour éviter de transformer du texte normal.
    if '\\"' in s or "\\n" in s or "\\t" in s:
        try:
            # decode escape sequences: \" -> ", \\n -> \n, etc.
            s = codecs.decode(s, "unicode_escape")
        except Exception:
            pass

    # 2) Retire guillemets d’encapsulation extrêmes
    s2 = _LEADING_BQ_RE.sub("", s)
    s2 = _TRAILING_BQ_RE.sub("", s2)
    if len(s2) < len(s) - 1:
        s = s2

    # 3) Normalisations simples
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _MULTI_NL_RE.sub("\n\n", s)
    s = _MULTI_SP_RE.sub(" ", s)

    return s.strip()

def noise_ratio(s: str) -> float:
    """
    Estime un ratio de "bruit" : part de caractères très atypiques.
    """
    if not s:
        return 1.0
    bad = _NOISE_CHARS_RE.findall(s)
    bad_len = sum(len(x) for x in bad)
    return bad_len / max(len(s), 1)


def is_too_boilerplate_or_too_short(s: str, min_chars: int, min_words: int) -> bool:
    if not s:
        return True
    if len(s) < min_chars:
        return True
    if len(s.split()) < min_words:
        return True
    return False


# ----------------------------
# Chunker sémantique
# ----------------------------

def dot_cos_sim_unit(a: np.ndarray, b: np.ndarray) -> float:
    # embeddings normalisés => cos = dot
    return float(np.dot(a, b))


class SemanticChunker:
    """
    Chunker sémantique par phrases, avec filtrage/fusion des micro-chunks.

    Paramètres principaux :
    - max_words : taille max approx (mots)
    - min_words_for_semantic_cut : taille min avant d'autoriser une coupure sémantique
    - sim_threshold : en dessous => rupture de sujet (si min_words atteint)
    - overlap_words : recouvrement approx (mots) via dernières phrases
    - min_chunk_words / min_chunk_chars : seuils de filtrage micro-chunks
    - max_noise_ratio : au-delà => chunk/texte suspect
    """
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        batch_size: int = 64,
        max_words: int = 450,
        min_words_for_semantic_cut: int = 140,
        sim_threshold: float = 0.58,
        overlap_words: int = 40,
        ema_alpha: float = 0.25,
        # filtrage post-chunk
        min_chunk_words: int = 80,
        min_chunk_chars: int = 400,
        max_noise_ratio: float = 0.06,
    ):
        ensure_nltk()
        model_name = os.environ.get("SECRETARIUS_SENTENCE_MODEL", model_name)
        local_only_env = os.environ.get("SECRETARIUS_LOCAL_FILES_ONLY", "1").strip().lower()
        local_files_only = local_only_env not in {"0", "false", "no"}
        try:
            self.model = SentenceTransformer(model_name, local_files_only=local_files_only)
        except TypeError:
            # Compatibility with older sentence-transformers versions.
            self.model = SentenceTransformer(model_name)
        self.batch_size = int(batch_size)

        self.max_words = int(max_words)
        self.min_words_for_semantic_cut = int(min_words_for_semantic_cut)
        self.sim_threshold = float(sim_threshold)
        self.overlap_words = int(overlap_words)
        self.ema_alpha = float(ema_alpha)

        self.min_chunk_words = int(min_chunk_words)
        self.min_chunk_chars = int(min_chunk_chars)
        self.max_noise_ratio = float(max_noise_ratio)

    @staticmethod
    def _wc(s: str) -> int:
        return len(s.split())

    def split_sentences(self, text: str) -> List[str]:
        try:
            sents = nltk.sent_tokenize(text, language="french")
        except Exception:
            sents = nltk.sent_tokenize(text)
        sents = [x.strip() for x in sents if x and x.strip()]
        # filtre très léger des phrases “vides” (ex. titres isolés)
        return sents

    def encode(self, sentences: List[str]) -> np.ndarray:
        embs = self.model.encode(
            sentences,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,  # important
            show_progress_bar=False,
        )
        return embs.astype(np.float32, copy=False)

    def _apply_overlap(self, cur_sentences: List[str]) -> List[str]:
        if self.overlap_words <= 0 or not cur_sentences:
            return []
        keep: List[str] = []
        keep_words = 0
        for s in reversed(cur_sentences):
            w = self._wc(s)
            if keep_words + w <= self.overlap_words:
                keep.insert(0, s)
                keep_words += w
            else:
                # force au moins 1 phrase si overlap demandé
                if not keep:
                    keep.insert(0, s)
                break
        return keep

    def chunk_raw(self, text: str) -> List[str]:
        """
        Chunking sémantique brut (sans filtrage final), sur texte déjà nettoyé.
        """
        sentences = self.split_sentences(text)
        if not sentences:
            return []

        embs = self.encode(sentences)

        chunks: List[str] = []
        cur_sentences: List[str] = []
        cur_words = 0
        cur_center: Optional[np.ndarray] = None

        def flush():
            nonlocal cur_sentences, cur_words, cur_center
            if cur_sentences:
                chunks.append(" ".join(cur_sentences).strip())

            overlap_sents = self._apply_overlap(cur_sentences)
            if overlap_sents:
                cur_sentences = overlap_sents
                cur_words = sum(self._wc(x) for x in cur_sentences)
                # recalcul centre (moyenne)
                o_embs = self.encode(cur_sentences)
                c = o_embs.mean(axis=0)
                n = np.linalg.norm(c)
                cur_center = (c / n) if n else None
            else:
                cur_sentences = []
                cur_words = 0
                cur_center = None

        for i, sent in enumerate(sentences):
            w = self._wc(sent)
            e = embs[i]

            if not cur_sentences:
                cur_sentences = [sent]
                cur_words = w
                cur_center = e.copy()
                continue

            sim = dot_cos_sim_unit(cur_center, e) if cur_center is not None else 1.0
            would_exceed = (cur_words + w > self.max_words)
            can_semantic_cut = (cur_words >= self.min_words_for_semantic_cut)
            semantic_break = (can_semantic_cut and sim < self.sim_threshold)

            if would_exceed or semantic_break:
                flush()
                # ajoute la phrase au nouveau chunk (avec overlap éventuel déjà dans cur_sentences)
                if not cur_sentences:
                    cur_sentences = [sent]
                    cur_words = w
                    cur_center = e.copy()
                else:
                    cur_sentences.append(sent)
                    cur_words += w
                    cur_center = (1.0 - self.ema_alpha) * cur_center + self.ema_alpha * e  # type: ignore
                    n = np.linalg.norm(cur_center)
                    if n:
                        cur_center = cur_center / n
                continue

            cur_sentences.append(sent)
            cur_words += w
            cur_center = (1.0 - self.ema_alpha) * cur_center + self.ema_alpha * e  # type: ignore
            n = np.linalg.norm(cur_center)
            if n:
                cur_center = cur_center / n

        if cur_sentences:
            chunks.append(" ".join(cur_sentences).strip())

        return chunks

    def _merge_too_small(self, chunks: List[str]) -> List[str]:
        """
        Fusionne les chunks trop courts avec le voisin suivant (ou précédent).
        """
        if not chunks:
            return []

        merged: List[str] = []
        buffer = ""

        def commit(buf: str):
            if buf.strip():
                merged.append(buf.strip())

        for ch in chunks:
            ch = ch.strip()
            if not ch:
                continue

            # Si buffer vide, init
            if not buffer:
                buffer = ch
                continue

            # Si buffer trop petit, on concatène avec ch
            if is_too_boilerplate_or_too_short(buffer, self.min_chunk_chars, self.min_chunk_words):
                buffer = (buffer + "\n\n" + ch).strip()
                continue

            # Sinon, on commit buffer et on repart
            commit(buffer)
            buffer = ch

        commit(buffer)

        # Deuxième passe : si le dernier est trop petit, le fusionner au précédent
        if len(merged) >= 2 and is_too_boilerplate_or_too_short(merged[-1], self.min_chunk_chars, self.min_chunk_words):
            merged[-2] = (merged[-2] + "\n\n" + merged[-1]).strip()
            merged.pop()

        return merged

    def _filter_noise(self, chunks: List[str]) -> List[str]:
        out: List[str] = []
        for ch in chunks:
            ch = ch.strip()
            if not ch:
                continue
            if noise_ratio(ch) > self.max_noise_ratio:
                # tentative: nettoyage léger supplémentaire
                ch2 = clean_text(ch)
                if noise_ratio(ch2) > self.max_noise_ratio:
                    continue
                ch = ch2
            out.append(ch)
        return out

    def chunk(self, raw_text: str) -> List[str]:
        """
        Pipeline complet : nettoyage -> chunking -> fusion micro-chunks -> filtrage bruit -> re-fusion si besoin.
        """
        text = clean_text(raw_text)

        # Filtrage très précoce : texte quasi vide ou très bruité
        if is_too_boilerplate_or_too_short(text, min_chars=200, min_words=40):
            return []
        if noise_ratio(text) > max(self.max_noise_ratio, 0.10):
            # si c'est très bruité, on tente un nettoyage simple et on re-teste
            text2 = clean_text(text)
            if noise_ratio(text2) > max(self.max_noise_ratio, 0.10):
                return []
            text = text2

        chunks = self.chunk_raw(text)
        chunks = self._merge_too_small(chunks)
        chunks = self._filter_noise(chunks)
        chunks = self._merge_too_small(chunks)  # une dernière passe après filtre
        return chunks


# ----------------------------
# I/O JSONL
# ----------------------------

def process_jsonl(
    input_path: str,
    output_path: str,
    chunker: SemanticChunker,
    keep_fields: Optional[List[str]] = None,
) -> Tuple[int, int, int]:
    """
    Retourne (docs_lus, docs_ecrits, total_chunks)
    """
    docs_in = 0
    docs_out = 0
    chunks_total = 0

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
        for line in tqdm(f_in, desc="Chunking"):
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            docs_in += 1
            contenu = rec.get("contenu", "")
            if isinstance(contenu, str) and contenu.startswith(("b\"", "b'")):
                contenu = clean_text(contenu)

            chunks = chunker.chunk(contenu)
            titre = " ".join(contenu.split()[:20]) if contenu else None

            out: Dict[str, Any] = {
                "source": rec.get("source"),
                "titre": titre,
                "chunks": chunks,
            }

            # Conserver d'autres champs si souhaité
            if keep_fields:
                for k in keep_fields:
                    if k in rec and k not in out and k != "contenu":
                        out[k] = rec[k]
            else:
                # défaut : conserver tout sauf contenu et chunks éventuels
                for k, v in rec.items():
                    if k in ("contenu", "chunks"):
                        continue
                    if k in out:
                        continue
                    out[k] = v

            f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
            docs_out += 1
            chunks_total += len(chunks)

    return docs_in, docs_out, chunks_total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="JSONL input (avec champ 'contenu')")
    p.add_argument("--output", required=True, help="JSONL output (avec champ 'chunks')")

    # Embeddings / modèle
    p.add_argument("--model", default="BAAI/bge-m3")
    p.add_argument("--batch-size", type=int, default=64)

    # Chunking sémantique
    p.add_argument("--max-words", type=int, default=450)
    p.add_argument("--min-words-for-semantic-cut", type=int, default=140)
    p.add_argument("--sim-threshold", type=float, default=0.58)
    p.add_argument("--overlap-words", type=int, default=40)
    p.add_argument("--ema-alpha", type=float, default=0.25)

    # Filtrage / fusion
    p.add_argument("--min-chunk-words", type=int, default=80)
    p.add_argument("--min-chunk-chars", type=int, default=400)
    p.add_argument("--max-noise-ratio", type=float, default=0.06)

    args = p.parse_args()

    chunker = SemanticChunker(
        model_name=args.model,
        batch_size=args.batch_size,
        max_words=args.max_words,
        min_words_for_semantic_cut=args.min_words_for_semantic_cut,
        sim_threshold=args.sim_threshold,
        overlap_words=args.overlap_words,
        ema_alpha=args.ema_alpha,
        min_chunk_words=args.min_chunk_words,
        min_chunk_chars=args.min_chunk_chars,
        max_noise_ratio=args.max_noise_ratio,
    )

    docs_in, docs_out, chunks_total = process_jsonl(args.input, args.output, chunker)
    avg = (chunks_total / docs_out) if docs_out else 0.0
    print(f"Docs lus: {docs_in} | Docs écrits: {docs_out} | Chunks total: {chunks_total} | Chunks/doc: {avg:.2f}")


if __name__ == "__main__":
    main()
