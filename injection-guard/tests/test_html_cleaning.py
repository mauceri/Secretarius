from injection_guard import clean_html


def test_removes_script_with_content():
    html = '<html><script>alert("xss"); steal()</script><p>Hello</p></html>'
    result = clean_html(html)
    assert 'alert' not in result
    assert 'steal' not in result
    assert 'Hello' in result


def test_removes_style_with_content():
    html = '<html><style>body { color: red; background: blue }</style><p>World</p></html>'
    result = clean_html(html)
    assert 'color' not in result
    assert 'World' in result


def test_removes_display_none():
    html = '<html><span style="display:none">HIDDEN</span><p>Visible</p></html>'
    result = clean_html(html)
    assert 'HIDDEN' not in result
    assert 'Visible' in result


def test_removes_visibility_hidden():
    html = '<html><div style="visibility:hidden">SECRET</div><p>OK</p></html>'
    result = clean_html(html)
    assert 'SECRET' not in result
    assert 'OK' in result


def test_removes_font_size_zero():
    html = '<html><span style="font-size:0">INVISIBLE</span><p>Text</p></html>'
    result = clean_html(html)
    assert 'INVISIBLE' not in result
    assert 'Text' in result


def test_removes_opacity_zero():
    html = '<html><div style="opacity:0">GHOST</div><p>Real</p></html>'
    result = clean_html(html)
    assert 'GHOST' not in result


def test_decodes_html_entities():
    html = '<p>AT&amp;T &lt;rocks&gt; &#x27;yes&#x27;</p>'
    result = clean_html(html)
    assert 'AT&T' in result
    assert '<rocks>' in result
    assert "'yes'" in result


def test_collapses_whitespace():
    html = '<p>Hello   \n  World</p>'
    result = clean_html(html)
    assert '  ' not in result
    assert '\n' not in result
    assert 'Hello' in result
    assert 'World' in result


def test_empty_input():
    assert clean_html('') == ''


def test_plain_text_no_html():
    text = 'No HTML here at all'
    result = clean_html(text)
    assert 'No HTML here at all' in result
