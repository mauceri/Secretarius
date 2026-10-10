# Courrier, étape 4 : relève de tiron@secretarius.co — plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** chaque minute, relever tiron@secretarius.co ; exécuter les commandes `/…` des messages qui viennent de l'utilisateur et sont authentifiés par Proton, réponse sur Telegram ; refuser les autres avec un avis Telegram.

**Architecture:** nouveau module `tiron_mail/releve.py` (authentification, extraction des commandes, boucle de relève), sous-commande `python3 -m tiron_mail releve`, lancée par un timer systemd utilisateur. Chaque commande est injectée par `openclaw agent --deliver --reply-channel telegram` : elle suit le même chemin qu'un message Telegram (derisk-deleg l'intercepte, vérifié le 2026-10-10). Le certificat du Bridge est épinglé par son empreinte SHA-256.

**Tech Stack:** Python 3.12 (imaplib, email), pytest, systemd utilisateur, OpenClaw 2026.8.2.

**Spec:** `docs/superpowers/specs/2026-10-07-agent-courrier-design.md`, § « Relève de tiron@secretarius.co », « Erreurs », « Tests ».

## Global Constraints

- Le corps d'un message refusé ne part jamais vers un modèle ; il n'apparaît pas dans l'avis.
- Un message ne peut pas être exécuté deux fois (journal des Message-ID, écrit **avant** l'exécution).
- Aucun secret dans le dépôt ; mots de passe par `secret_env` (`Wiki_LM/.env` pour le service).
- Pas de `systemctl enable/start` sans l'accord de l'utilisateur.
- Chemin complet d'openclaw (`OPENCLAW_BIN` ou `~/.local/bin/openclaw`) : le PATH systemd est minimal.

## Règle d'authentification (vérifiée le 2026-10-10 sur messages réels)

Un message est accepté si **toutes** ces conditions tiennent :

1. un seul `From`, dans `adresses_proprietaire`, et `Return-Path` identique ;
2. soit `X-Pm-Origin: internal` **et** `X-Pm-Content-Encryption: end-to-end` (envoi depuis votre compte Proton : pas d'`Authentication-Results`) ;
3. soit `X-Pm-Origin: external` **et** au moins un `Authentication-Results` de `mail.protonmail.ch` avec `dmarc=pass`, aucun avec un autre résultat dmarc.

Proton remplace les `X-Pm-*` fournis par l'expéditeur (essai du 2026-10-10 : `internal` falsifié reçu `external`). Écart à la spec, voulu : le repli « spf=pass et dkim=pass » est retiré, car il ne garantit pas l'alignement avec le `From`.

## Review Focus

- Message de l'utilisateur dont l'objet seul porte la commande, corps HTML (Proton envoie du HTML) → commande trouvée.
- Message transféré ou cité : les lignes citées commencent par `>` → pas exécutées.
- Commande en échec (openclaw arrêté) → avis Telegram, message rangé dans Traite, pas réexécuté.
- Bridge arrêté dix minutes → un seul avis.
- Message sans Message-ID → refusé.

---

### Task 1 : épinglage du certificat du Bridge

**Files:**
- Modify: `tiron_mail/config.py` (champ `empreinte_tls`)
- Modify: `tiron_mail/boite.py` (`verifier_empreinte`, appel dans `_connecter_imap`)
- Modify: `tiron_mail/envoi.py` (appel dans `_smtp`)
- Test: `tiron_mail/tests/test_boite.py`

**Interfaces:**
- Produces: `Compte.empreinte_tls: str = ""` ; `verifier_empreinte(sock, compte) -> None` (lève `ErreurBoite`).

- [ ] **Step 1 : test**

```python
# tiron_mail/tests/test_boite.py (ajout)
import hashlib
from tiron_mail.boite import verifier_empreinte


class _Sock:
    def __init__(self, der): self.der = der
    def getpeercert(self, binary_form=False): return self.der


def test_empreinte_conforme_acceptee(compte_gmail):
    emp = hashlib.sha256(b"cert").hexdigest().upper()
    compte = replace(compte_gmail, empreinte_tls=":".join(emp[i:i + 2] for i in range(0, 64, 2)))
    verifier_empreinte(_Sock(b"cert"), compte)


def test_empreinte_differente_refusee(compte_gmail):
    compte = replace(compte_gmail, empreinte_tls=hashlib.sha256(b"cert").hexdigest())
    with pytest.raises(ErreurBoite, match="certificat"):
        verifier_empreinte(_Sock(b"autre"), compte)


def test_sans_empreinte_pas_de_controle(compte_gmail):
    verifier_empreinte(_Sock(b"importe"), compte_gmail)
```

- [ ] **Step 2 :** `python -m pytest tiron_mail/tests/test_boite.py -q` → FAIL (`ImportError: verifier_empreinte`).

- [ ] **Step 3 : implémentation**

`config.py`, dans `Compte`, après `tri_par_date` :

```python
    # Empreinte SHA-256 du certificat du serveur (Proton Bridge, auto-signé) :
    # « openssl s_client -starttls imap -connect 127.0.0.1:1143 | openssl x509 -noout -fingerprint -sha256 ».
    empreinte_tls: str = ""
```

`boite.py` (imports `hashlib`, `hmac`) :

```python
def verifier_empreinte(sock, compte: Compte) -> None:
    """Le Bridge a un certificat auto-signé : sans vérification par autorité, on
    compare son empreinte à celle notée dans la configuration."""
    if not compte.empreinte_tls:
        return
    vue = hashlib.sha256(sock.getpeercert(binary_form=True) or b"").hexdigest()
    attendue = compte.empreinte_tls.replace(":", "").lower()
    if not hmac.compare_digest(vue, attendue):
        raise ErreurBoite(f"boîte {compte.nom} : certificat inattendu ({vue[:16]}…), connexion refusée")
```

Dans `_connecter_imap`, avant chaque `return conn` : `verifier_empreinte(conn.sock, compte)`. Dans `envoi._smtp`, idem (import depuis `tiron_mail.boite`).

- [ ] **Step 4 :** `python -m pytest tiron_mail -q` → tout passe.
- [ ] **Step 5 :** `git commit -m "feat(tiron_mail): épinglage du certificat du Bridge"`

### Task 2 : authentification et commandes

**Files:**
- Create: `tiron_mail/releve.py`
- Test: `tiron_mail/tests/test_releve.py`

**Interfaces:**
- Produces: `authentifier(m: email.message.Message, proprietaires: tuple) -> str` ("" si accepté, sinon le motif) ; `commandes(objet: str, texte: str) -> list[str]`.

- [ ] **Step 1 : tests**

```python
# tiron_mail/tests/test_releve.py
import email
import email.policy

from tiron_mail.releve import authentifier, commandes

PROPRIO = ("christian@mauceri.me", "cmauceri@gmail.com")


def msg(entetes: list[tuple[str, str]]):
    brut = "".join(f"{k}: {v}\r\n" for k, v in entetes) + "\r\ncorps\r\n"
    return email.message_from_string(brut, policy=email.policy.default)


INTERNE = [("From", "C M <christian@mauceri.me>"), ("Return-Path", "<christian@mauceri.me>"),
           ("X-Pm-Origin", "internal"), ("X-Pm-Content-Encryption", "end-to-end")]
EXTERNE = [("From", "C M <cmauceri@gmail.com>"), ("Return-Path", "<cmauceri@gmail.com>"),
           ("X-Pm-Origin", "external"), ("X-Pm-Content-Encryption", "on-delivery"),
           ("Authentication-Results", "mail.protonmail.ch; dmarc=pass (p=none dis=none) header.from=gmail.com"),
           ("Authentication-Results", "mail.protonmail.ch; spf=pass smtp.mailfrom=gmail.com")]


def test_interne_du_proprietaire_accepte():
    assert authentifier(msg(INTERNE), PROPRIO) == ""


def test_externe_dmarc_pass_accepte():
    assert authentifier(msg(EXTERNE), PROPRIO) == ""


def test_expediteur_inconnu_refuse():
    m = msg([("From", "x@ex.fr"), ("Return-Path", "<x@ex.fr>")] + INTERNE[2:])
    assert "non autorisé" in authentifier(m, PROPRIO)


def test_return_path_different_refuse():
    m = msg([INTERNE[0], ("Return-Path", "<x@ex.fr>")] + INTERNE[2:])
    assert "Return-Path" in authentifier(m, PROPRIO)


def test_interne_non_chiffre_refuse():
    m = msg(INTERNE[:3] + [("X-Pm-Content-Encryption", "on-delivery")])
    assert authentifier(m, PROPRIO)


def test_externe_dmarc_fail_refuse_meme_avec_un_pass_ajoute():
    m = msg(EXTERNE + [("Authentication-Results", "mail.protonmail.ch; dmarc=fail header.from=gmail.com")])
    assert "DMARC" in authentifier(m, PROPRIO)


def test_externe_dmarc_d_un_autre_serveur_ignore():
    m = msg(EXTERNE[:4] + [("Authentication-Results", "mx.pirate.ex; dmarc=pass")])
    assert "DMARC" in authentifier(m, PROPRIO)


def test_origine_absente_refusee():
    assert authentifier(msg(INTERNE[:2]), PROPRIO)


def test_deux_from_refuses():
    m = msg([("From", "christian@mauceri.me, x@ex.fr")] + INTERNE[1:])
    assert authentifier(m, PROPRIO)


def test_commandes_objet_et_lignes_du_corps():
    assert commandes("/wikistatus", "Bonjour\n/inbox 3\n> /ecrire cité\n  /r chiens") == \
        ["/wikistatus", "/inbox 3", "/r chiens"]


def test_sans_commande():
    assert commandes("wikistatus", "pas de barre") == []
```

- [ ] **Step 2 :** `python -m pytest tiron_mail/tests/test_releve.py -q` → FAIL (module absent).

- [ ] **Step 3 : implémentation**

```python
"""Relève de tiron@secretarius.co : n'exécute que les commandes des messages de
l'utilisateur authentifiés par Proton (spec courrier, § Relève). Le corps d'un
message refusé n'est jamais transmis."""
from __future__ import annotations

import re
from email.message import Message
from email.utils import getaddresses, parseaddr

_DMARC = re.compile(r"\bdmarc=(\w+)")


def authentifier(m: Message, proprietaires: tuple) -> str:
    """"" si le message vient de l'utilisateur et que Proton l'authentifie, sinon le motif."""
    froms = getaddresses([str(v) for v in m.get_all("From", [])])
    if len(froms) != 1:
        return "expéditeur absent ou multiple"
    exp = froms[0][1].lower()
    if exp not in proprietaires:
        return f"expéditeur non autorisé ({exp})"
    if parseaddr(str(m.get("Return-Path", "")))[1].lower() != exp:
        return "Return-Path différent de l'expéditeur"
    origine = [str(v).strip() for v in m.get_all("X-Pm-Origin", [])]
    chiffrement = [str(v).strip() for v in m.get_all("X-Pm-Content-Encryption", [])]
    if origine == ["internal"]:
        return "" if chiffrement == ["end-to-end"] else "message interne non chiffré de bout en bout"
    if origine != ["external"]:
        return "origine Proton inconnue"
    dmarc = [r.group(1) for v in m.get_all("Authentication-Results", [])
             if str(v).strip().startswith("mail.protonmail.ch;") and (r := _DMARC.search(str(v)))]
    if not dmarc or any(d != "pass" for d in dmarc):
        return "DMARC non validé par Proton"
    return ""


def commandes(objet: str, texte: str) -> list[str]:
    """Objet et lignes du corps qui commencent par « / » (les lignes citées commencent par « > »)."""
    return [l.strip() for l in [objet, *texte.splitlines()] if l.strip().startswith("/")]
```

- [ ] **Step 4 :** `python -m pytest tiron_mail -q` → tout passe.
- [ ] **Step 5 :** `git commit -m "feat(tiron_mail): authentification Proton et commandes de la relève"`

### Task 3 : boucle de relève et sous-commande `releve`

**Files:**
- Modify: `tiron_mail/config.py` (`Config.releve`)
- Modify: `tiron_mail/boite.py` (`_ouvrir(lecture_seule=True)`)
- Modify: `tiron_mail/releve.py` (`relever`, `executer_openclaw`)
- Modify: `tiron_mail/__main__.py` (sous-commande `releve`)
- Modify: `tiron_mail/tests/conftest.py` (`create`, `uid move` dans `FauxIMAP`)
- Test: `tiron_mail/tests/test_releve.py`

**Interfaces:**
- Consumes: `authentifier`, `commandes` (Task 2) ; `notifier_telegram(env)` de `tiron_cron.livraison`.
- Produces: `relever(cfg, env, executer, notifier, etat: Path, connecter=_connecter_imap) -> str`.

Comportement :
- Dossiers `Folders/Traite` et `Folders/Refuse` (ASCII : pas d'encodage UTF-7 modifié), créés s'ils manquent (le `NO` d'un dossier existant est ignoré).
- `select INBOX` en écriture ; le `FETCH (RFC822)` sans PEEK marque le message lu.
- Message déjà au journal → rangé dans Traite sans exécution.
- Refusé (motif d'`authentifier`, ou pas de Message-ID) → avis « Courriel refusé sur tiron@… : de <exp>, <motif>. Rien n'a été exécuté. », rangé dans Refuse.
- Accepté → Message-ID au journal (écrit sur disque) **puis** exécution de chaque commande ; sans commande, avis « écrivez une ligne qui commence par / » ; une commande en échec donne un avis et n'arrête pas les suivantes ; rangé dans Traite.
- Connexion impossible → compteur d'échecs ; avis unique au dixième ; remis à zéro au premier succès.
- État : `~/.local/state/tiron_mail/releve.json` = `{"traites": [...], "echecs": n}`.

- [ ] **Step 1 : faux serveur**

Dans `FauxIMAP.__init__` : `self.ranges: list = []` ; ajouter

```python
    def create(self, dossier):
        return "OK", [b""]
```

et, dans `uid`, avant le `raise` : `if commande == "move": self.ranges.append((int(args[0]), args[1])); self.messages.pop(int(args[0])); return "OK", [b""]`.

- [ ] **Step 2 : tests**

```python
# tiron_mail/tests/test_releve.py (ajout)
from email.message import EmailMessage

import pytest

from tiron_mail.config import Config
from tiron_mail.releve import relever
from tiron_mail.tests.conftest import FauxIMAP


def brut_releve(n, objet, entetes, corps="Bonjour"):
    m = EmailMessage()
    for k, v in entetes:
        m[k] = v
    m["Subject"] = objet
    m["Message-ID"] = f"<r{n}@ex.fr>"
    m.set_content(corps)
    return m.as_bytes()


@pytest.fixture
def cfg(compte_gmail):
    return Config({"tiron": compte_gmail}, "tiron", PROPRIO, releve="tiron")


def lancer(cfg, serveur, tmp_path, executer=None):
    faits, avis = [], []
    texte = relever(cfg, {"GMAIL_APP_PASSWORD": "x"}, executer or faits.append, avis.append,
                    tmp_path / "etat.json", connecter=lambda c: serveur)
    return faits, avis, texte


def test_commande_executee_puis_rangee(cfg, tmp_path):
    serveur = FauxIMAP({1: brut_releve(1, "/wikistatus", INTERNE)})
    faits, avis, _ = lancer(cfg, serveur, tmp_path)
    assert faits == ["/wikistatus"] and avis == []
    assert serveur.ranges == [(1, "Folders/Traite")]


def test_refus_avis_sans_le_corps(cfg, tmp_path):
    entetes = [("From", "x@ex.fr"), ("Return-Path", "<x@ex.fr>")] + INTERNE[2:]
    serveur = FauxIMAP({1: brut_releve(1, "/inbox", entetes, corps="SECRET-DU-CORPS")})
    faits, avis, _ = lancer(cfg, serveur, tmp_path)
    assert faits == [] and serveur.ranges == [(1, "Folders/Refuse")]
    assert "x@ex.fr" in avis[0] and "SECRET-DU-CORPS" not in avis[0]


def test_jamais_deux_fois(cfg, tmp_path):
    lancer(cfg, FauxIMAP({1: brut_releve(1, "/wikistatus", INTERNE)}), tmp_path)
    serveur = FauxIMAP({7: brut_releve(1, "/wikistatus", INTERNE)})  # même Message-ID
    faits, _, _ = lancer(cfg, serveur, tmp_path)
    assert faits == [] and serveur.ranges == [(7, "Folders/Traite")]


def test_sans_commande_avis(cfg, tmp_path):
    faits, avis, _ = lancer(cfg, FauxIMAP({1: brut_releve(1, "wikistatus", INTERNE)}), tmp_path)
    assert faits == [] and "/" in avis[0]


def test_commande_en_echec_avis_et_suite(cfg, tmp_path):
    def executer(c):
        if c == "/a":
            raise OSError("openclaw arrêté")
    serveur = FauxIMAP({1: brut_releve(1, "/a", INTERNE, corps="/b")})
    _, avis, _ = lancer(cfg, serveur, tmp_path, executer=executer)
    assert "/a" in avis[0] and serveur.ranges == [(1, "Folders/Traite")]


def test_sans_message_id_refuse(cfg, tmp_path):
    m = EmailMessage()
    for k, v in INTERNE:
        m[k] = v
    m["Subject"] = "/wikistatus"
    m.set_content("x")
    serveur = FauxIMAP({1: m.as_bytes()})
    faits, avis, _ = lancer(cfg, serveur, tmp_path)
    assert faits == [] and "Message-ID" in avis[0]


def test_dix_echecs_un_seul_avis(cfg, tmp_path):
    def refus(c):
        raise OSError("Bridge arrêté")
    avis = []
    for _ in range(12):
        with pytest.raises(Exception):
            relever(cfg, {"GMAIL_APP_PASSWORD": "x"}, lambda c: None, avis.append,
                    tmp_path / "etat.json", connecter=refus)
    assert len(avis) == 1
```

- [ ] **Step 3 :** `python -m pytest tiron_mail/tests/test_releve.py -q` → FAIL (`relever` absent, `Config` sans `releve`).

- [ ] **Step 4 : implémentation**

`config.py` : `Config` gagne `releve: str = ""` (dernier champ) ; dans `charger` :

```python
    releve = d.get("releve", {}).get("boite", "")
    if releve and releve not in comptes:
        raise ErreurConfig(f"releve.boite « {releve} » ne désigne aucune boîte configurée")
    return Config(comptes, defaut, tuple(a.lower() for a in envoi.get("adresses_proprietaire", [])), releve)
```

`boite.py` : `def _ouvrir(self, lecture_seule: bool = True)` et `conn.select("INBOX", readonly=lecture_seule)`.

`releve.py` (imports à compléter : `email`, `email.policy`, `json`, `subprocess`, `Path`, `Mapping`, `Boite`, `ErreurBoite`, `_connecter_imap`, `analyser`, `ErreurConfig`) :

```python
TRAITE, REFUSE = "Folders/Traite", "Folders/Refuse"
ETAT_DEFAUT = Path.home() / ".local" / "state" / "tiron_mail" / "releve.json"
_SEUIL_ECHECS = 10


def _lire_etat(chemin: Path) -> dict:
    try:
        return json.loads(chemin.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"traites": [], "echecs": 0}


def _ecrire_etat(chemin: Path, etat: dict) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    tmp = chemin.with_suffix(".tmp")
    tmp.write_text(json.dumps(etat), encoding="utf-8")
    tmp.replace(chemin)


def relever(cfg, env, executer, notifier, etat_chemin: Path = ETAT_DEFAUT, connecter=_connecter_imap) -> str:
    if not cfg.releve:
        raise ErreurConfig("aucune boîte à relever ([releve] boite = …)")
    compte = cfg.comptes[cfg.releve]
    etat = _lire_etat(etat_chemin)
    b = Boite(compte, env, connecter)
    try:
        conn = b._ouvrir(lecture_seule=False)
    except ErreurBoite as e:
        etat["echecs"] = etat.get("echecs", 0) + 1
        _ecrire_etat(etat_chemin, etat)
        if etat["echecs"] == _SEUIL_ECHECS:
            notifier(f"Relève de {compte.adresse} en échec depuis {_SEUIL_ECHECS} essais : {e}")
        raise
    etat["echecs"] = 0
    bilan: list[str] = []
    try:
        for dossier in (TRAITE, REFUSE):
            conn.create(dossier)  # « NO » si le dossier existe : sans effet
        typ, data = conn.uid("search", None, "ALL")
        for u in (data[0] or b"").split() if typ == "OK" else []:
            brut = b._fetch(conn, int(u), "(RFC822)")
            m = email.message_from_bytes(brut, policy=email.policy.default)
            mid = str(m.get("Message-ID", "")).strip()
            exp = parseaddr(str(m.get("From", "")))[1]
            if mid and mid in etat["traites"]:
                conn.uid("move", u.decode(), TRAITE)
                continue
            motif = authentifier(m, cfg.adresses_proprietaire) or ("" if mid else "message sans Message-ID")
            if motif:
                notifier(f"Courriel refusé sur {compte.adresse} : de {exp}, {motif}. Rien n'a été exécuté.")
                conn.uid("move", u.decode(), REFUSE)
                bilan.append(f"refusé : {exp} ({motif})")
                continue
            etat["traites"].append(mid)
            _ecrire_etat(etat_chemin, etat)  # avant l'exécution : jamais deux fois
            a = analyser(brut)
            cmds = commandes(a["objet"], a["texte"])
            if not cmds:
                notifier(f"Courriel de {exp} reçu sur {compte.adresse} sans commande : "
                         "écrivez une ligne qui commence par /, par exemple /wikistatus.")
            for c in cmds:
                try:
                    executer(c)
                except Exception as e:  # les commandes suivantes passent quand même
                    notifier(f"Commande {c} (courriel de {exp}) en échec : {e}")
            conn.uid("move", u.decode(), TRAITE)
            bilan.append(f"{exp} : {len(cmds)} commande(s)")
    finally:
        _ecrire_etat(etat_chemin, etat)
        try:
            conn.logout()
        except Exception:
            pass
    return "\n".join(bilan) or "Rien à relever."


def executer_openclaw(env: Mapping[str, str]):
    """Injecte la commande comme un message Telegram ; la réponse part sur Telegram."""
    def executer(commande: str) -> None:
        exe = env.get("OPENCLAW_BIN") or str(Path.home() / ".local" / "bin" / "openclaw")
        subprocess.run([exe, "agent", "--agent", "main", "--session-key", "agent:main:courrier",
                        "--message", commande, "--deliver", "--reply-channel", "telegram",
                        "--reply-to", env["TIRON_TELEGRAM_CHAT_ID"], "--json"],
                       capture_output=True, text=True, timeout=600, check=True)
    return executer
```

Le test des dix échecs passe par `_ouvrir`, qui transforme l'`OSError` du connecteur en `ErreurBoite`.

`__main__.py` : ajouter `"releve"` aux `choices` et à la docstring (`releve   relève de tiron@ (service tiron-mail-releve)`), puis la branche :

```python
        elif a.commande == "releve":
            from tiron_cron.livraison import notifier_telegram
            from tiron_mail.releve import executer_openclaw, relever
            out = {"ok": True, "text": relever(cfg, env, executer_openclaw(env), notifier_telegram(env),
                                               connecter=connecter)}
```

- [ ] **Step 5 :** `python -m pytest tiron_mail tiron_cron -q` → tout passe.
- [ ] **Step 6 :** `git commit -m "feat(tiron_mail): relève de tiron@ (journal, rangement, avis)"`

### Task 4 : service, configuration, documentation

**Files:**
- Create: `openclaw-config/tiron-mail-releve.service`, `openclaw-config/tiron-mail-releve.timer`
- Modify: `openclaw-config/install.sh` (copie des deux unités, sans activation)
- Modify: `tiron_mail/comptes.toml.modele` (`[releve]`, `empreinte_tls`)
- Modify: `docs/components/obsidian.md` ou doc courrier existante : section « Relève de tiron@ »
- Local (hors dépôt) : `~/.config/tiron_mail/comptes.toml`

- [ ] **Step 1 : unités**

```ini
# tiron-mail-releve.service
[Unit]
Description=Tiron — relève de tiron@secretarius.co (commandes de l'utilisateur)
After=network.target

[Service]
Type=oneshot
WorkingDirectory=%h/Secretarius
EnvironmentFile=%h/Secretarius/Wiki_LM/.env
ExecStart=/usr/bin/python3 -m tiron_mail releve
StandardOutput=journal
StandardError=journal
```

```ini
# tiron-mail-releve.timer
[Unit]
Description=Relève de tiron@secretarius.co (chaque minute)

[Timer]
OnCalendar=*:*:00

[Install]
WantedBy=timers.target
```

Dans `install.sh`, ajouter les deux fichiers à la boucle de copie des unités, à côté de `wiki-lm-embed` ; pas d'`enable`.

- [ ] **Step 2 : configuration locale** (sauvegarde `.bak-20261010-releve`) : ajouter `[releve]\nboite = "tiron"` et, aux cinq boîtes Proton, `empreinte_tls = "A8:6F:77:59:…:A7:BD"` (relevée le 2026-10-10 sur 1143 ; vérifier que 1026 présente la même). Même ajout, commenté, dans le modèle.

- [ ] **Step 3 : essai à blanc** : `python3 -m tiron_mail inbox --args "1 tiron"` (l'épinglage ne casse rien), puis `python3 -m pytest tiron_mail tiron_cron -q`.

- [ ] **Step 4 :** documentation, puis `git commit -m "feat(tiron_mail): service de relève de tiron@ ; doc"`.

### Task 5 : essai réel (avec l'utilisateur)

Les six messages déjà dans la boîte de réception de tiron@ seront traités au premier passage (avis « sans commande » ou refus). Proposer d'abord de les archiver à la main.

1. L'utilisateur installe et active le timer : `cp openclaw-config/tiron-mail-releve.* ~/.config/systemd/user/ && systemctl --user daemon-reload && systemctl --user enable --now tiron-mail-releve.timer`.
2. Depuis christian@mauceri.me : objet `/wikistatus` → réponse Telegram en moins de deux minutes, message dans Traite.
3. Depuis Gmail : objet quelconque, corps `/inbox 2 tiron` → réponse Telegram.
4. Depuis contact@secretarius.co (adresse non autorisée) : `/wikistatus` → avis de refus, rien d'exécuté, message dans Refuse.
5. Renvoi du message 2 tel quel (même Message-ID impossible depuis l'interface : vérifier plutôt que le journal contient les Message-ID).
6. `journalctl --user -u tiron-mail-releve -n 20` : pas d'erreur.
