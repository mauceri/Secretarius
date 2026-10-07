"""Construction et envoi d'un message par SMTP. L'envoi n'est jamais décidé
ici : la ligne de commande n'envoie qu'avec --confirme (après /confirm)."""
from __future__ import annotations

import smtplib
from collections.abc import Mapping
from email.message import EmailMessage
from email.utils import formatdate, make_msgid

from tiron_mail.boite import ErreurBoite, _contexte_tls
from tiron_mail.config import Compte


def construire(depuis: Compte, a: str, objet: str, corps: str, en_reponse_a: str | None = None) -> EmailMessage:
    m = EmailMessage()
    m["From"] = depuis.adresse
    m["To"] = a
    m["Subject"] = objet
    m["Date"] = formatdate(localtime=True)
    m["Message-ID"] = make_msgid(domain=depuis.adresse.split("@")[-1])
    if en_reponse_a:
        m["In-Reply-To"] = en_reponse_a
        m["References"] = en_reponse_a
    m.set_content(corps)
    return m


def _smtp(compte: Compte):
    if compte.securite_smtp == "ssl":
        return smtplib.SMTP_SSL(compte.hote_smtp, compte.port_smtp,
                                context=_contexte_tls(compte.hote_smtp), timeout=30)
    conn = smtplib.SMTP(compte.hote_smtp, compte.port_smtp, timeout=30)
    conn.starttls(context=_contexte_tls(compte.hote_smtp))
    return conn


def envoyer(msg: EmailMessage, compte: Compte, env: Mapping[str, str], smtp=_smtp) -> None:
    mdp = env.get(compte.secret_env, "")
    if not mdp:
        raise ErreurBoite(f"boîte {compte.nom} : mot de passe absent ({compte.secret_env})")
    try:
        conn = smtp(compte)
        conn.login(compte.utilisateur, mdp)
        conn.send_message(msg)
        conn.quit()
    except (OSError, smtplib.SMTPException) as e:
        raise ErreurBoite(f"envoi refusé par {compte.nom} : {e}") from e
