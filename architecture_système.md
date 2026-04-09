# Assistants IA : l'Alternative Locale, Frugale et Souveraine

**Une architecture innovante pour les professionnels indépendants**

*Christian Mauceri — cmauceri@gmail.com — Avril 2026*

---

## Pourquoi une IA locale et souveraine ?

![](./Architecture_R2D2.png)

---

## Le problème des assistants IA classiques

- ☁️ **Vos données partent dans le cloud** — chez OpenAI, Google, Microsoft…
- 💸 **Coûts imprévisibles** — abonnements + surcoûts selon l'usage
- 🔗 **Dépendance totale** — si le service change, vous subissez
- 🔓 **Risques de confidentialité** — vos conversations servent à entraîner les modèles

> *Pour un agent immobilier, un avocat, un comptable : vos données clients méritent mieux.*

---

## L'alternative : OpenClaw / R2D2

Un assistant IA **installé chez vous** (ou sur votre serveur privé), qui :

- 🏠 Garde vos données **sur votre infrastructure**
- 💬 Répond sur **Telegram** — simple, immédiat, sans nouvelle app
- 📧 Lit vos mails, rédige des réponses, gère votre agenda
- 💶 Coûte entre **5 € et 60 € par mois** — prévisible, maîtrisé

---

## L'architecture deux agents

```
        Vous  ──Telegram──▶  R2D2 (agent principal)
                                    │
                             fichiers JSON
                                    │
                                    ▼
                            Scout (agent isolé)
                          lecture web uniquement
```

**R2D2** applique vos règles et prend les décisions.

**Scout** lit le web et les documents externes — **sans pouvoir agir.**

---

## Pourquoi deux agents ?

Quand R2D2 doit lire une page web ou un mail externe, il délègue à Scout.

Scout est **physiquement incapable** de :

- ❌ Envoyer un mail
- ❌ Modifier un fichier hors de sa zone
- ❌ Contacter Telegram ou Google
- ❌ Lancer une commande

Si une page contient une tentative de manipulation, **elle reste sans effet.**

---

## Ce que R2D2 sait faire

- 📬 **Mails** — lire, résumer, rédiger, envoyer (via Google)
- 📅 **Agenda** — consulter, créer des événements
- 📄 **Documents** — créer des notes, des comptes-rendus dans Obsidian
- 🌐 **Recherches web** — via Scout, en toute sécurité
- 📊 **Présentations** — générer des fichiers PowerPoint via Pandoc

---

## Les modèles IA disponibles

| Modèle | Hébergement | Coût | TEE |
|---|---|---|---|
| DeepSeek Chat | API cloud | ~0,30 €/M tokens | Non |
| DeepSeek V3.1 | OLLM/NEAR | Prépayé | ✅ Oui |
| Gemma 4 / GLM 4 | Local (sanroque) | 0 € | ✅ Oui |

**TEE** = Trusted Execution Environment — attestation cryptographique que personne ne voit vos données, même le fournisseur.

---

## Le modèle économique

| Poste | Coût mensuel |
|---|---|
| VPS Hetzner (santiago) | ~5 € |
| API DeepSeek (usage normal) | 5 à 20 € |
| OLLM DeepSeek V3.1 (TEE) | Prépayé à l'usage |
| Modèles locaux (sanroque) | 0 € |
| **Total** | **5 € à 60 € / mois** |

*Zéro abonnement opaque. Zéro surprise.*

---

## En résumé

> Un assistant IA **intelligent, frugal et local** —
> sous votre contrôle, pour votre activité.

- ✅ Vos données restent chez vous
- ✅ Coût fixe et prévisible
- ✅ Protection contre les injections de prompt
- ✅ Adaptable à votre métier

---

## Prochaines étapes

- 🎬 Démonstration en conditions réelles
- 🔎 Analyse de vos usages et besoins
- 🛠️ Déploiement sur mesure

**Contact :** cmauceri@gmail.com
