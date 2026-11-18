# session-broker

Transport Session Messenger de Secretarius (spec `docs/superpowers/specs/2026-11-18-canal-session-design.md`).
Écoute sur 127.0.0.1, jeton `TIRON_SESSION_BROKER_TOKEN` ; seul le plugin de canal `session` du gateway s'y connecte.

1. `cd ~/Secretarius/session-broker && ~/.bun/bin/bun install --frozen-lockfile`
2. `~/.bun/bin/bun src/main.ts init` : ajoute la phrase secrète et le jeton à `~/.config/secrets.env`, affiche l'identifiant Session du bot.
3. `~/.config/session-broker/config.json` : `{"port": 8091, "allowFrom": ["<votre identifiant Session, 66 caractères>"]}`
4. `cp systemd/session-broker.service ~/.config/systemd/user/ && systemctl --user daemon-reload`
5. `systemctl --user enable --now session-broker`
6. Vérifier : `GET /health` avec le jeton → `ok: true`.
7. Veille (alerte par courriel si Tiron ne reçoit plus) : `avis = "<votre adresse>"` dans
   `[envoi]` de `~/.config/tiron_mail/comptes.toml`, puis
   `cp systemd/session-broker-veille.{service,timer} ~/.config/systemd/user/ && systemctl --user daemon-reload && systemctl --user enable --now session-broker-veille.timer`.
8. Rappels `/cron` sur Session : `TIRON_SESSION_ID=<votre identifiant Session>` dans `~/.config/secrets.env`, puis `/maj`.
