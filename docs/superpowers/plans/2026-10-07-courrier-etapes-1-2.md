# Courrier, étapes 1 et 2 : interblocage et `tiron_mail` — plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal :** rendre les sous-agents de nouveau utilisables (gog : agenda et Drive), puis remplacer gog pour le courrier par un module IMAP/SMTP maison, branché sur les commandes Telegram `/inbox`, `/chercher`, `/lire`, `/repondre` et `/ecrire`.

**Architecture :** `tiron_mail` (Python standard, ligne de commande JSON) suit le modèle de `tiron_cron`. derisk-deleg l'appelle directement, sans sous-agent. Les boîtes sont décrites dans `~/.config/tiron_mail/comptes.toml`, sans secret ; les mots de passe sont dans `~/.config/secrets.env`, transmis au gateway. Gmail est branché d'abord. Les deux boîtes Proton suivent dès que Proton Bridge est installé (tâche 7, avec l'utilisateur).

**Tech Stack :** Python 3.12 (`imaplib`, `smtplib`, `email`, `tomllib`), pytest ; TypeScript, vitest (derisk-deleg) ; OpenClaw 2026.8.2 ; Proton Bridge (CLI).

**Spec :** `docs/superpowers/specs/2026-10-07-agent-courrier-design.md`

## Global Constraints

- gog ne sert plus au courrier ; il garde l'agenda et Drive.
- Corps d'un message lu = contenu non fiable : texte seul, pièces jointes listées mais jamais ouvertes, rien n'est transmis à un modèle comme instruction.
- Règle d'envoi appliquée dans `tiron_mail` : sans `--confirme`, rien ne part ; avec `--confirme`, envoi. Tout envoi depuis Telegram passe par aperçu puis `/confirm`.
- Réponse depuis la boîte qui a reçu le message ; `/ecrire` depuis le compte `envoi.defaut` (Proton perso ; Gmail tant que le Bridge n'est pas installé).
- Identifiants courts : `<préfixe>:<uid IMAP>` (`g:` Gmail, `p:` Proton perso, `t:` Tiron).
- Aucun secret dans le dépôt ni dans `comptes.toml`.
- Redémarrage de services et modification de `~/.openclaw/openclaw.json` : avec l'accord de l'utilisateur.
- Tests : `cd ~/Secretarius && python3 -m pytest tiron_mail -q` ; `cd derisk-deleg && npx vitest run` ; `LAYA_CHECKPOINT=x python3 -m pytest router_service -q`.

## Review Focus

- **Corps HTML seulement** (beaucoup de lettres d'information) : converti en texte lisible, balises et scripts retirés, jamais exécuté → test dans la tâche 3.
- **Objet encodé (RFC 2047, accents)** : décodé (« Réunion » et non `=?utf-8?q?R=C3=A9union?=`) → test dans la tâche 3.
- **Identifiant d'une boîte non configurée** (`p:12` avant l'installation du Bridge) : message clair, pas d'exception → test dans la tâche 4.
- **Une boîte injoignable** (Bridge arrêté) pendant `/inbox` sur toutes les boîtes : les autres boîtes s'affichent, la boîte en erreur est signalée → test dans la tâche 4.
- **`/repondre` puis `/confirm` après plus de 10 minutes** : rien ne part → couvert par le mécanisme `pending` existant, test dans la tâche 6.

---

### Task 1: interblocage de la voie `main` (essai, puis réglage)

**Files :** aucun dans le dépôt pour l'essai ; `openclaw-config/openclaw.json.template` si le réglage est retenu.

- [ ] **Step 1 : comprendre la voie.** Lire la documentation d'OpenClaw sur les voies d'exécution (`openclaw docs` ou docs.openclaw.ai : « lanes », `maxConcurrent`, `subagents.maxConcurrent`). Relever dans le journal (`journalctl --user -u openclaw-gateway --since 2026-10-01 | grep "lane wait exceeded"`) les voies concernées et les dates. Vérifier si les `/source` (Scout) sont aussi touchés.
- [ ] **Step 2 : [accord] réglage.** Hypothèse principale : `agents.defaults.maxConcurrent: 1` empêche le sous-agent de démarrer tant que le tour en cours, qui l'attend, occupe la voie. Réglage :
  ```bash
  cp ~/.openclaw/openclaw.json ~/.openclaw/openclaw.json.bak-$(date +%Y%m%d-%H%M%S)
  openclaw config set agents.defaults.maxConcurrent 2
  openclaw config validate
  ```
  Si la documentation (step 1) désigne une autre cause, appliquer la correction qu'elle indique et le consigner dans le registre.
- [ ] **Step 3 : vérifier.** Sur Telegram, avec l'utilisateur : `/drive budget` (gog) répond en moins de 60 s, sans « (run timeout) ». Le journal ne montre plus `lane wait exceeded` pendant l'essai.
- [ ] **Step 4 : modèle de configuration.** Reporter la valeur dans `openclaw-config/openclaw.json.template` (`"maxConcurrent": 2`), commit : `fix(openclaw): deux tâches sur la voie principale — les sous-agents (gog, scout) n'attendent plus le tour qui les lance`.

---

### Task 2: `tiron_mail/config.py` — comptes

**Files :**
- Create: `tiron_mail/__init__.py` (vide), `tiron_mail/tests/__init__.py` (vide), `tiron_mail/config.py`
- Create: `tiron_mail/comptes.toml.modele`
- Test: `tiron_mail/tests/test_config.py`

**Interfaces :**
- Produces : `@dataclass(frozen=True) Compte(nom, prefixe, adresse, utilisateur, hote_imap, port_imap, securite_imap, hote_smtp, port_smtp, securite_smtp, secret_env)` (`securite_*` ∈ `"ssl"`, `"starttls"`) ; `@dataclass(frozen=True) Config(comptes: dict[str, Compte], envoi_defaut: str, adresses_proprietaire: tuple[str, ...])` ; `charger(chemin: Path) -> Config` ; `compte_de_prefixe(cfg, prefixe) -> Compte` (lève `ErreurConfig`) ; `class ErreurConfig(ValueError)`.

- [ ] **Step 1 : tests**

```python
# tiron_mail/tests/test_config.py
import pytest

from tiron_mail.config import ErreurConfig, charger, compte_de_prefixe

TOML = """
[envoi]
defaut = "gmail"
adresses_proprietaire = ["moi@gmail.com", "moi@proton.me"]

[comptes.gmail]
prefixe = "g"
adresse = "moi@gmail.com"
utilisateur = "moi@gmail.com"
hote_imap = "imap.gmail.com"
port_imap = 993
securite_imap = "ssl"
hote_smtp = "smtp.gmail.com"
port_smtp = 465
securite_smtp = "ssl"
secret_env = "GMAIL_APP_PASSWORD"
"""


def test_charger(tmp_path):
    p = tmp_path / "comptes.toml"
    p.write_text(TOML, encoding="utf-8")
    cfg = charger(p)
    assert cfg.envoi_defaut == "gmail"
    assert cfg.adresses_proprietaire == ("moi@gmail.com", "moi@proton.me")
    g = cfg.comptes["gmail"]
    assert g.prefixe == "g" and g.port_imap == 993 and g.secret_env == "GMAIL_APP_PASSWORD"


def test_prefixe_inconnu(tmp_path):
    p = tmp_path / "comptes.toml"
    p.write_text(TOML, encoding="utf-8")
    with pytest.raises(ErreurConfig, match="aucune boîte configurée pour « p »"):
        compte_de_prefixe(charger(p), "p")


def test_envoi_defaut_doit_exister(tmp_path):
    p = tmp_path / "comptes.toml"
    p.write_text(TOML.replace('defaut = "gmail"', 'defaut = "proton"'), encoding="utf-8")
    with pytest.raises(ErreurConfig, match="envoi.defaut"):
        charger(p)


def test_fichier_absent(tmp_path):
    with pytest.raises(ErreurConfig, match="introuvable"):
        charger(tmp_path / "absent.toml")


def test_modele_du_depot_valide():
    from pathlib import Path
    cfg = charger(Path(__file__).parent.parent / "comptes.toml.modele")
    assert set(cfg.comptes) == {"gmail", "proton", "tiron"}
```

- [ ] **Step 2 : échec** — `python3 -m pytest tiron_mail -q` → FAIL.

- [ ] **Step 3 : implémentation**

```python
# tiron_mail/config.py
"""Description des boîtes (sans secret) : ~/.config/tiron_mail/comptes.toml.
Les mots de passe restent dans l'environnement (secret_env), cf. spec 2026-10-07."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

CHEMIN_DEFAUT = Path.home() / ".config" / "tiron_mail" / "comptes.toml"


class ErreurConfig(ValueError):
    pass


@dataclass(frozen=True)
class Compte:
    nom: str
    prefixe: str
    adresse: str
    utilisateur: str
    hote_imap: str
    port_imap: int
    securite_imap: str
    hote_smtp: str
    port_smtp: int
    securite_smtp: str
    secret_env: str


@dataclass(frozen=True)
class Config:
    comptes: dict
    envoi_defaut: str
    adresses_proprietaire: tuple


def charger(chemin: Path = CHEMIN_DEFAUT) -> Config:
    if not Path(chemin).exists():
        raise ErreurConfig(f"configuration des boîtes introuvable : {chemin}")
    d = tomllib.loads(Path(chemin).read_text(encoding="utf-8"))
    comptes = {nom: Compte(nom=nom, **c) for nom, c in d.get("comptes", {}).items()}
    envoi = d.get("envoi", {})
    defaut = envoi.get("defaut", "")
    if defaut not in comptes:
        raise ErreurConfig(f"envoi.defaut « {defaut} » ne désigne aucune boîte configurée")
    return Config(comptes, defaut, tuple(a.lower() for a in envoi.get("adresses_proprietaire", [])))


def compte_de_prefixe(cfg: Config, prefixe: str) -> Compte:
    for c in cfg.comptes.values():
        if c.prefixe == prefixe:
            return c
    raise ErreurConfig(f"aucune boîte configurée pour « {prefixe} »")
```

`tiron_mail/comptes.toml.modele` : les trois boîtes. Gmail : `imap.gmail.com:993` ssl, `smtp.gmail.com:465` ssl, `GMAIL_APP_PASSWORD`. Proton perso : `127.0.0.1:1143` starttls, `127.0.0.1:1025` starttls, `PROTON_BRIDGE_PASS_PERSO`. Tiron : idem, `PROTON_BRIDGE_PASS_TIRON`. Adresses de l'utilisateur en exemple (`vous@gmail.com`…), `envoi.defaut = "proton"`, chaque section commentée en français.

- [ ] **Step 4 : vérifier** → PASS. **Step 5 : commit** `feat(mail): configuration des boîtes (tiron_mail.config)`.

---

### Task 3: `tiron_mail/message.py` — lecture d'un message

**Files :** Create `tiron_mail/message.py` ; Test `tiron_mail/tests/test_message.py`

**Interfaces :**
- Produces : `analyser(brut: bytes) -> dict` (`expediteur`, `destinataires`, `objet`, `date`, `message_id`, `texte`, `pieces_jointes: list[str]`) ; `resume(m: dict, ident: str) -> str` (une ligne : identifiant, date courte, expéditeur, objet).

- [ ] **Step 1 : tests**

```python
# tiron_mail/tests/test_message.py
from email.message import EmailMessage

from tiron_mail.message import analyser, resume


def _brut(**kw):
    m = EmailMessage()
    m["From"] = kw.get("de", "Paul <paul@ex.fr>")
    m["To"] = "moi@gmail.com"
    m["Subject"] = kw.get("objet", "Chantier")
    m["Date"] = "Tue, 06 Oct 2026 09:12:00 +0200"
    m["Message-ID"] = "<abc@ex.fr>"
    if "html" in kw:
        m.set_content(kw["html"], subtype="html")
    else:
        m.set_content(kw.get("texte", "Bonjour,\nle chantier avance."))
    if kw.get("pj"):
        m.add_attachment(b"%PDF", maintype="application", subtype="pdf", filename="devis.pdf")
    return m.as_bytes()


def test_message_texte():
    m = analyser(_brut())
    assert m["expediteur"] == "Paul <paul@ex.fr>" and m["objet"] == "Chantier"
    assert "le chantier avance" in m["texte"] and m["message_id"] == "<abc@ex.fr>"


def test_objet_encode_decode():
    m = analyser(_brut(objet="Réunion de lundi"))
    assert m["objet"] == "Réunion de lundi"


def test_html_converti_en_texte_sans_script():
    m = analyser(_brut(html="<html><body><p>Lettre <b>n°5</b></p><script>alert(1)</script></body></html>"))
    assert "Lettre n°5" in m["texte"] and "alert" not in m["texte"] and "<" not in m["texte"]


def test_pieces_jointes_listees_pas_ouvertes():
    m = analyser(_brut(pj=True))
    assert m["pieces_jointes"] == ["devis.pdf"] and "%PDF" not in m["texte"]


def test_resume():
    assert resume(analyser(_brut()), "g:42") == "g:42  06/10 09:12  Paul <paul@ex.fr>  Chantier"
```

- [ ] **Step 2 : échec.** **Step 3 : implémentation** avec `email.message_from_bytes(brut, policy=email.policy.default)`. `get_body(preferencelist=("plain", "html"))` ; pour le HTML, un `html.parser.HTMLParser` qui ignore `script` et `style` et rassemble le texte (espaces normalisés). Pièces jointes : `iter_attachments()`, nom seulement. `resume` : date par `email.utils.parsedate_to_datetime`, au format `%d/%m %H:%M`.
- [ ] **Step 4 : vérifier → PASS. Step 5 : commit** `feat(mail): lecture d'un message (texte seul, pièces jointes listées)`.

---

### Task 4: `tiron_mail/boite.py` — IMAP (lister, chercher, lire)

**Files :** Create `tiron_mail/boite.py` ; Test `tiron_mail/tests/test_boite.py`

**Interfaces :**
- Consumes : tâches 2, 3.
- Produces : `class Boite(compte, env, connecter=_connecter_imap)` avec `lister(n=10) -> list[dict]`, `chercher(criteres: str, n=20) -> list[dict]` (chaque dict : `id`, `resume`), `lire(uid: str) -> dict` ; `criteres_imap(texte: str, aujourd_hui: date) -> list[str]` ; `class ErreurBoite(RuntimeError)`.

- [ ] **Step 1 : tests** — un faux client IMAP (classe de test) qui implémente `login`, `select`, `uid("search", …)`, `uid("fetch", …)`, `logout`, avec trois messages construits par `_brut` (tâche 3).

```python
# tiron_mail/tests/test_boite.py (extraits essentiels)
from datetime import date

import pytest

from tiron_mail.boite import Boite, ErreurBoite, criteres_imap


def test_criteres_imap():
    j = date(2026, 10, 7)
    assert criteres_imap("de:paul chantier", j) == ["FROM", '"paul"', "TEXT", '"chantier"']
    assert criteres_imap("objet:devis depuis:7j", j) == ["SUBJECT", '"devis"', "SINCE", "30-Sep-2026"]
    assert criteres_imap("", j) == ["ALL"]


def test_lister_les_plus_recents_d_abord(faux_imap, compte_gmail):
    b = Boite(compte_gmail, {"GMAIL_APP_PASSWORD": "x"}, connecter=faux_imap)
    ids = [m["id"] for m in b.lister(2)]
    assert ids == ["g:3", "g:2"]


def test_lire(faux_imap, compte_gmail):
    m = Boite(compte_gmail, {"GMAIL_APP_PASSWORD": "x"}, connecter=faux_imap).lire("2")
    assert m["objet"] == "Message 2"


def test_mot_de_passe_absent(faux_imap, compte_gmail):
    with pytest.raises(ErreurBoite, match="GMAIL_APP_PASSWORD"):
        Boite(compte_gmail, {}, connecter=faux_imap).lister()


def test_boite_injoignable(compte_gmail):
    def refus(compte):
        raise OSError("connexion refusée")
    with pytest.raises(ErreurBoite, match="injoignable"):
        Boite(compte_gmail, {"GMAIL_APP_PASSWORD": "x"}, connecter=refus).lister()


def test_uid_introuvable(faux_imap, compte_gmail):
    with pytest.raises(ErreurBoite, match="introuvable"):
        Boite(compte_gmail, {"GMAIL_APP_PASSWORD": "x"}, connecter=faux_imap).lire("99")
```

Les fixtures `faux_imap` et `compte_gmail` vont dans `tiron_mail/tests/conftest.py`. `faux_imap` renvoie, pour `uid("search", None, *crit)`, les UID `b"1 2 3"`, et pour `uid("fetch", "2", "(RFC822)")` le message brut correspondant ; `uid("fetch", "99", …)` renvoie `("OK", [None])`.

- [ ] **Step 2 : échec.** **Step 3 : implémentation**
  - `_connecter_imap(compte)` : `imaplib.IMAP4_SSL(hote, port)` si `ssl`, sinon `IMAP4(hote, port)` puis `starttls(ssl_context)`. Pour Proton Bridge sur `127.0.0.1`, le contexte TLS accepte son certificat auto-signé. Exception explicite et commentée, limitée à l'hôte local.
  - `lister` : `select("INBOX", readonly=True)`, `uid("search", None, "ALL")`, puis les n derniers UID en ordre décroissant, `fetch (RFC822.HEADER)` pour le résumé.
  - `chercher` : `criteres_imap` (mots-clés `de:`, `objet:`, `depuis:Nj` ; le reste va en `TEXT`).
  - `lire` : `fetch (RFC822)` puis `analyser`.
  - Toute exception réseau ou d'authentification devient `ErreurBoite("boîte <nom> injoignable : …")`.
- [ ] **Step 4 : vérifier → PASS. Step 5 : commit** `feat(mail): accès IMAP (lister, chercher, lire)`.

---

### Task 5: `tiron_mail/envoi.py` et `tiron_mail/__main__.py` — SMTP et ligne de commande

**Files :** Create `tiron_mail/envoi.py`, `tiron_mail/__main__.py` ; Test `tiron_mail/tests/test_envoi.py`, `tiron_mail/tests/test_cli.py`

**Interfaces :**
- Produces (contrat pour derisk-deleg) : ligne de commande `python3 -m tiron_mail <sous-commande>`, une ligne JSON `{"ok": bool, "text": str, ...}` en sortie, code 0.
  - `inbox --args "<n> <compte>"`
  - `chercher --args "<critères> [compte]"`
  - `lire --args "<id>"`
  - `repondre --args "<id> <texte>" [--confirme]`
  - `ecrire --args "<adresse> <texte>" [--confirme]`
- Sans `--confirme`, `repondre` et `ecrire` renvoient l'aperçu (`text`) et `args` normalisés. Avec `--confirme`, ils envoient.
- `inbox` et `chercher` sans compte parcourent toutes les boîtes. Une boîte en erreur est signalée par une ligne « boîte <nom> injoignable : … », et les autres sont affichées.

- [ ] **Step 1 : tests** (faux SMTP qui enregistre les messages ; faux IMAP de la tâche 4)

```python
# tiron_mail/tests/test_cli.py (extraits essentiels)
def test_repondre_sans_confirme_n_envoie_rien(capsys, env_test, faux_smtp):
    out = _call(capsys, ["repondre", "--args", "g:2 D'accord pour jeudi"], faux_smtp=faux_smtp)
    assert out["ok"] and "Réponse à « Message 2 »" in out["text"] and faux_smtp.envoyes == []


def test_repondre_confirme_envoie_depuis_la_boite_d_origine(capsys, env_test, faux_smtp):
    _call(capsys, ["repondre", "--args", "g:2 D'accord pour jeudi", "--confirme"], faux_smtp=faux_smtp)
    m = faux_smtp.envoyes[0]
    assert m["From"] == "moi@gmail.com" and m["Subject"] == "Re: Message 2"
    assert m["In-Reply-To"] == "<m2@ex.fr>"


def test_ecrire_depuis_le_compte_par_defaut(capsys, env_test, faux_smtp):
    _call(capsys, ["ecrire", "--args", "marie@ex.fr Compte rendu ci-dessous", "--confirme"], faux_smtp=faux_smtp)
    assert faux_smtp.envoyes[0]["From"] == "moi@gmail.com"


def test_inbox_une_boite_en_erreur_n_empeche_pas_les_autres(capsys, env_deux_boites):
    out = _call(capsys, ["inbox", "--args", ""])
    assert "boîte proton injoignable" in out["text"] and "g:3" in out["text"]


def test_lire_boite_non_configuree(capsys, env_test):
    out = _call(capsys, ["lire", "--args", "p:12"])
    assert out["ok"] is False and "aucune boîte configurée pour « p »" in out["text"]


def test_lire_marque_le_contenu_non_fiable(capsys, env_test):
    out = _call(capsys, ["lire", "--args", "g:2"])
    assert out["text"].startswith("Contenu du message (non fiable, à lire seulement) :")
```

`env_test` : configuration temporaire (une boîte Gmail), `TIRON_MAIL_CONFIG` pointant dessus, faux IMAP injecté. `_call` appelle `main(argv, connecter=..., smtp=...)`.

- [ ] **Step 2 : échec.** **Step 3 : implémentation**
  - `envoi.py` : `construire(depuis: Compte, a, objet, corps, en_reponse_a=None) -> EmailMessage`, qui pose `In-Reply-To` et `References` pour une réponse. `envoyer(msg, compte, env, smtp=_smtp)` : `SMTP_SSL` ou `SMTP` plus `starttls`, `login`, `send_message`. Erreurs converties en `ErreurBoite`.
  - `__main__.py` : analyse des `--args`, comme `tiron_cron` ; `TIRON_MAIL_CONFIG` (par défaut `~/.config/tiron_mail/comptes.toml`).
- [ ] **Step 4 : vérifier** — `python3 -m pytest tiron_mail -q` → PASS (toute la suite).
- [ ] **Step 5 : commit** `feat(mail): envoi SMTP et ligne de commande tiron_mail`.

---

### Task 6: derisk-deleg et routeur — commandes courrier sans gog

**Files :**
- Create: `derisk-deleg/src/mail-ops.ts`, `derisk-deleg/src/mail-ops.test.ts`
- Modify: `derisk-deleg/src/dispatch.ts`, `derisk-deleg/src/index.ts`, `router_service/router.py`, `router_service/server.py`
- Test: `derisk-deleg/src/dispatch.test.ts`, `derisk-deleg/src/index.test.ts`, `router_service/test_router.py`

**Interfaces :**
- Consumes : ligne de commande de la tâche 5.
- Produces : `runTironMail(argv: string[]): Promise<{ ok: boolean; text: string; args?: string }>`, sur le modèle de `runTironCron` (même `execFile` asynchrone, même délai) ; `commandToAction` : `/inbox`, `/chercher`, `/lire` → `{kind: "mail", op}` ; `/repondre`, `/ecrire` → `{kind: "mail-envoi", op}` ; `/drive`, `/connecter` restent `gog`. Routeur : `/ecrire` reconnue tapée (`NEEDS_ARG`).

- [ ] **Step 1 : tests vitest**
  - `/inbox` appelle `runTironMail(["inbox", "--args", ""])` et relaie le texte, sans `api.runtime.subagent.run`.
  - `/repondre g:2 ok` donne l'aperçu, puis `/confirm` appelle `runTironMail(["repondre", "--args", "g:2 ok", "--confirme"])`.
  - `/annuler` n'envoie rien.
  - `/confirm` après 11 minutes n'envoie rien.
  - `/ecrire marie@ex.fr texte` suit le même parcours.
  - `/drive budget` passe toujours par gog.

  Côté pytest du routeur : `/ecrire` tapée est honorée telle quelle.
- [ ] **Step 2 : échec. Step 3 : implémentation**
  - `mail-ops.ts`, sur le modèle de `cron-ops.ts`.
  - Nouveau type de `pending` : `{ kind: "mail"; op: "repondre" | "ecrire"; args; ts }`, géré dans `/confirm` et `/annuler` comme `cron`.
  - Les anciennes branches `gog-reply` et `parseReply` sont retirées, puisque `/repondre` ne passe plus par gog. `parse.ts` et son test sont supprimés s'ils ne servent plus.
- [ ] **Step 4 : vérifier** — `npm test && npm run build` ; `LAYA_CHECKPOINT=x python3 -m pytest router_service -q` → PASS.
- [ ] **Step 5 : commit** `feat(derisk-deleg): courrier par tiron_mail (IMAP/SMTP), plus par gog`.

---

### Task 7: mise en service (**avec l'utilisateur**)

- [ ] **Gmail** : l'utilisateur crée un mot de passe d'application Google (compte Google, Sécurité, « Mots de passe des applications », double authentification requise). Il l'ajoute à `~/.config/secrets.env` sous le nom `GMAIL_APP_PASSWORD`, puis lance `/maj` pour le propager au gateway.
- [ ] Créer `~/.config/tiron_mail/comptes.toml` à partir du modèle, avec la seule boîte Gmail et `envoi.defaut = "gmail"` en attendant le Bridge.
- [ ] **[accord]** Réinstaller derisk-deleg dans `~/.openclaw/extensions/`, puis redémarrer `openclaw-gateway` et `tiron-router`.
- [ ] Essais Telegram : `/inbox`, `/chercher de:<un expéditeur réel>`, `/lire g:<id>`, `/repondre g:<id> essai` → `/annuler`, puis une réponse réelle à soi-même avec `/confirm`.
- [ ] **Proton Bridge** (session dédiée, avec l'utilisateur) :
  - installer la version en ligne de commande, avec un trousseau `pass` ;
  - `bridge --cli`, puis `login` pour le Proton perso et pour tiron@secretarius.co (double authentification) ;
  - relever les mots de passe locaux (`info`) et les placer dans `secrets.env` (`PROTON_BRIDGE_PASS_PERSO`, `PROTON_BRIDGE_PASS_TIRON`) ;
  - créer un service systemd utilisateur pour Bridge.

  Ensuite : boîtes `proton` et `tiron` dans `comptes.toml`, `envoi.defaut = "proton"`, et mêmes essais avec `p:`.
- [ ] Mettre à jour `faits.md` : commandes courrier, et gog réservé à l'agenda et à Drive.
