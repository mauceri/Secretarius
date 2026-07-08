#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Métriques du harnais de validation secretarius (fonctions pures)."""

LABELS = ["wiki", "gog", "secretarius", "null"]


def confusion_matrix(pairs):
    m = {t: {p: 0 for p in LABELS} for t in LABELS}
    for vrai, pred in pairs:
        m[vrai][pred] += 1
    return m


def taux_commandes_volees(m):
    total = sum(m["wiki"].values()) + sum(m["gog"].values())
    if total == 0:
        return 0.0
    volees = m["wiki"]["secretarius"] + m["gog"]["secretarius"]
    return volees / total


def rappel(m, label):
    total = sum(m[label].values())
    if total == 0:
        return 0.0
    return m[label][label] / total
