# Courrier (tiron_mail)

Spec : `docs/superpowers/specs/2026-10-07-agent-courrier-design.md`. Boîtes décrites dans
`~/.config/tiron_mail/comptes.toml` (modèle : `tiron_mail/comptes.toml.modele`), mots de
passe dans l'environnement. Commandes Telegram (`/inbox`, `/lire`, `/repondre`, `/ecrire`…)
par derisk-deleg ; blocs `mail?`/`mail!` d'Obsidian : voir `obsidian.md`.

## Relève de tiron@secretarius.co

Chaque minute, `tiron-mail-releve.timer` lance `python3 -m tiron_mail releve`.

- **Accepté** seulement si le message vient de vous, avec ces trois conditions :
  - `From` unique, présent dans `adresses_proprietaire`, et `Return-Path` identique ;
  - envoi depuis votre compte Proton : `X-Pm-Origin: internal` et
    `X-Pm-Content-Encryption: end-to-end` ;
  - envoi depuis l'extérieur (Gmail) : `X-Pm-Origin: external` et `dmarc=pass` posé par
    `mail.protonmail.ch` (tout autre résultat dmarc entraîne un refus).

  Proton remplace les `X-Pm-*` fournis par l'expéditeur (vérifié le 2026-10-10).
- **Commandes** : l'objet s'il commence par `/`, puis les lignes `/` en tête du corps,
  jusqu'à la première autre ligne. Un message transféré ou cité n'est donc jamais
  exécuté. `/confirm`, `/annuler` et `/maj` sont ignorés (avis) : ils se tapent sur
  Telegram seulement. Chaque commande est injectée par `openclaw agent --session-key agent:main:courrier --deliver` : même
  chemin qu'un message Telegram ; la réponse arrive sur Telegram, et `/confirm` se tape
  sur Telegram.
- **Refusé** : rien n'est exécuté, avis Telegram (expéditeur, motif), corps jamais transmis.
- **Une seule fois** : le Message-ID est noté avant l'exécution dans
  `~/.local/state/tiron_mail/releve.json`. Le message est rangé dans `Folders/Traite` ou
  `Folders/Refuse`, et marqué lu.
- **Mise en service** (journal absent ou illisible) : les messages déjà présents sont
  rangés dans Traite sans être exécutés, avec un avis.
- **Pannes** : une commande en échec donne un avis et n'arrête pas les suivantes. Dix
  échecs de connexion de suite donnent un seul avis. Un avis qui échoue n'arrête pas la
  relève (il est écrit dans le journal systemd).
- **Certificat du Bridge épinglé** : `empreinte_tls` des boîtes Proton (même certificat
  sur 1143 et 1026). Après une réinstallation du Bridge, relever la nouvelle empreinte :
  `openssl s_client -starttls imap -connect 127.0.0.1:1143 </dev/null | openssl x509 -noout -fingerprint -sha256`.

Activation (une fois) :
`systemctl --user daemon-reload && systemctl --user enable --now tiron-mail-releve.timer`.
Journal : `journalctl --user -u tiron-mail-releve -n 20`.
