"""
Patch ponctuel : reformate les sections ## Extrait Wikipedia dans les pages wiki.

- Convertit les listes wikicode (séparées par ";") en tirets Markdown
- Fixe les YAML malformés (titres non quotés avec ":")
- Ne touche pas au contenu LLM, uniquement à la section Wikipedia et au frontmatter

Usage :
    python tools/patch_wiki_abstracts.py              # dry-run
    python tools/patch_wiki_abstracts.py --apply      # écriture effective
    python tools/patch_wiki_abstracts.py --apply --wiki /chemin/vers/wiki
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import frontmatter


def _format_abstract(abstract: str) -> str:
    # Pages d'homonymie Wikipedia : items séparés par " ;\n"
    if " ;\n" in abstract or (abstract.rstrip().endswith(" ;") and "\n" in abstract):
        # Séparer intro (avant ":") du reste si l'intro est courte
        m = re.match(r"^(.{0,120}[:\uff1a])\s*(.*)", abstract, re.DOTALL)
        intro, body = (m.group(1).strip(), m.group(2)) if m else ("", abstract)
        items = [i.strip().rstrip(";.").strip()
                 for i in re.split(r"(?<=[;.])\s*\n", body)]
        items = [i for i in items if i]
        result = (intro + "\n\n") if intro else ""
        result += "\n".join("- " + i for i in items)
        return result
    # Listes wikicode : ";" en début de ligne
    lines = []
    for line in abstract.splitlines():
        s = line.strip()
        lines.append(("- " + s[1:].strip()) if s.startswith(";") else s)
    return "\n".join(l for l in lines if l)


_SECTION_RE = re.compile(
    r"(## Extrait Wikipedia\n\n)(.*?)(\n\*\[Source Wikipedia\])",
    re.DOTALL,
)


def patch_file(path: Path, apply: bool) -> bool:
    """Retourne True si une modification est nécessaire."""
    text = path.read_text(encoding="utf-8")

    changed = False

    # 1. Reformater l'abstract Wikipedia
    def _replace_abstract(m: re.Match) -> str:
        nonlocal changed
        original = m.group(2)
        fixed = _format_abstract(original)
        if fixed != original:
            changed = True
        return m.group(1) + fixed + m.group(3)

    new_text = _SECTION_RE.sub(_replace_abstract, text)

    # 2. Fixer le YAML (titres non quotés)
    try:
        post = frontmatter.loads(new_text)
        serialized = frontmatter.dumps(post)
        if serialized != new_text:
            changed = True
            new_text = serialized
    except Exception:
        pass

    if changed and apply:
        path.write_text(new_text, encoding="utf-8")

    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Écrire les modifications")
    parser.add_argument(
        "--wiki",
        default=os.environ.get(
            "WIKI_PATH",
            str(Path.home() / "Documents/Arbath/Wiki_LM")
        ),
    )
    args = parser.parse_args()

    wiki_dir = Path(args.wiki) / "wiki"
    pages = sorted(p for p in wiki_dir.glob("*.md")
                   if any(p.stem.startswith(x) for x in ("c-", "e-")))

    patched = skipped = 0
    for page in pages:
        if patch_file(page, apply=args.apply):
            patched += 1
            if not args.apply:
                print(f"  [dry] {page.name}")
        else:
            skipped += 1

    action = "Patchées" if args.apply else "À patcher"
    print(f"\n{action} : {patched} page(s) — inchangées : {skipped}")
    if not args.apply:
        print("Relancez avec --apply pour écrire.")


if __name__ == "__main__":
    main()
