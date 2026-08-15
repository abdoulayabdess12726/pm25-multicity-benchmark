#!/usr/bin/env python3
"""
scripts/regenerate_figures.py — squelette (P4).

La régénération complète des 5 figures (vectoriel PDF+SVG, annotations,
bandes de confiance) est spécifiée en détail dans P10 (typographie/palette
cohérentes, Figure 4 à corriger spécifiquement) — HORS PÉRIMÈTRE de la tâche
P4, qui ne détaillait que les tables et leurs assertions.

Ce script existe pour que `python scripts/regenerate_figures.py` ne soit
jamais une commande inconnue, mais ne génère rien de silencieux : il
s'arrête explicitement plutôt que de produire des figures non spécifiées.
"""
import sys

if __name__ == "__main__":
    print("regenerate_figures.py n'est pas encore implémenté — la spécification "
          "complète des figures (P10) n'a pas été fournie dans cette tâche (P4). "
          "Rien n'a été généré silencieusement.", file=sys.stderr)
    sys.exit(1)
