import unittest

from secretarius_markdown import (
    SecretariusMarkdownError,
    parse_secretarius_markdown,
    render_secretarius_command,
)


class TestSecretariusMarkdown(unittest.TestCase):
    def test_parse_returns_none_without_block(self):
        self.assertIsNone(parse_secretarius_markdown("# note libre"))

    def test_render_index_command_from_block_and_body(self):
        parsed = parse_secretarius_markdown(
            """# Cavalerie rouge

```secretarius
action: index
doc_id: doc:boudienny-001
type_note: lecture
title: Cavalerie rouge
tags: URSS, cavalerie
```

Texte de la note...
"""
        )
        command = render_secretarius_command(parsed)
        self.assertEqual(
            command,
            "/index\n"
            "doc_id: doc:boudienny-001\n"
            "type_note: lecture\n"
            "title: Cavalerie rouge\n"
            "#URSS #cavalerie\n"
            "# Cavalerie rouge\n\nTexte de la note...",
        )

    def test_render_req_command_requires_query(self):
        parsed = parse_secretarius_markdown(
            """```secretarius
action: req
query: cavalerie rouge URSS
```"""
        )
        self.assertEqual(render_secretarius_command(parsed), "/req cavalerie rouge URSS")

    def test_render_update_requires_doc_id(self):
        parsed = parse_secretarius_markdown(
            """```secretarius
action: update
```

Texte corrige..."""
        )
        with self.assertRaises(SecretariusMarkdownError):
            render_secretarius_command(parsed)

    def test_multiple_blocks_raise_error(self):
        with self.assertRaises(SecretariusMarkdownError):
            parse_secretarius_markdown(
                """```secretarius
action: req
query: a
```

```secretarius
action: req
query: b
```"""
            )


if __name__ == "__main__":
    unittest.main()
