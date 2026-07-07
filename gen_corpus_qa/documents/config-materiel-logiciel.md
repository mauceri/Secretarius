# Configuration matérielle et logicielle de Secretarius (machine sanroque)

## Matériel
- Machine : sanroque, ordinateur portable.
- Processeur : AMD Ryzen 9 6900HX.
- Carte graphique intégrée (iGPU) : AMD Radeon 680M (architecture RDNA2, identifiant gfx1035).
- Mémoire vive : 30 Go, partagée entre le processeur et l'iGPU.

## Modèle de langage
- Le modèle qui anime Tiron est Phi-4-mini-instruct, quantifié en Q6_K, augmenté d'adaptateurs LoRA spécialisés.
- L'extraction d'expressions du wiki utilise un second modèle : Phi-4-mini affiné sur Wikipédia en français.

## Services actifs (systemd utilisateur)
- slm-llama_cpp : serveur llama.cpp sur le port 8998, sert Phi-4-mini + l'adaptateur de routage, accéléré par ROCm.
- tiron-router : service de routage sur le port 8999, classe le message et sélectionne la commande.
- openclaw-gateway : passerelle reliée à Telegram, exécute Tiron.
- llama.cpp extracteur : port 8989, modèle Wikipédia FR.
