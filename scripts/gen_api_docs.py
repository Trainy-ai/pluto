#!/usr/bin/env python3
"""Generate Mintlify-ready MDX for Pluto's public Python API.

This is the *producer* half of a pull-based docs pipeline. It parses the
``pluto`` package **statically** with griffe (no runtime imports, so it does
not need ``httpx``/``numpy``/``torch`` installed) and emits one ``.mdx`` file
per page plus a ``meta.json`` manifest into ``docs-api/``.

The Mintlify docs repo (``Trainy-ai/konduktor``) consumes ``docs-api/`` from
this repo at whatever tag it pins, and builds its own ``docs.json`` navigation
from ``meta.json``. The only cross-repo contract is the output directory, the
stable per-page filenames, and the ``meta.json`` schema below.

Run::

    python scripts/gen_api_docs.py            # writes docs-api/
    python scripts/gen_api_docs.py --check     # fail if output is stale (CI)

The doc set is driven by the PAGES manifest, which is anchored to
``pluto.__all__``. Add a symbol to ``__all__`` and to PAGES and it shows up;
internal modules never leak in.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import griffe

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / 'docs-api'

# meta.json schema version. Bump when the consumer-facing shape changes so the
# konduktor side can guard against it.
META_VERSION = 1


@dataclass
class Page:
    """One output ``.mdx`` page.

    slug:     output filename (``<slug>.mdx``) and Mintlify page id.
    title:    page title (frontmatter + nav label).
    group:    suggested Mintlify nav group; the consumer may regroup freely.
    symbols:  griffe dotted paths (relative to the package) to render, in order.
    intro:    optional markdown rendered under the title.
    """

    slug: str
    title: str
    group: str
    symbols: List[str]
    intro: str = ''


# ---------------------------------------------------------------------------
# Page manifest. Anchored to pluto.__all__ (see pluto/__init__.py).
#
# Runtime-bound names: pluto.log / pluto.watch / pluto.alert are assigned from
# Op methods at runtime (op.py: `pluto.log, pluto.alert, pluto.watch = ...`),
# so we document them from their real source `op.Op.*`.
# ---------------------------------------------------------------------------
PAGES: List[Page] = [
    Page(
        slug='initialization',
        title='Initialization & lifecycle',
        group='Core',
        intro='Start, configure, and finish a Pluto run.',
        symbols=[
            'init.init',
            'init.finish',
            'sets.setup',
            'auth.login',
            'auth.logout',
            'util.generate_run_id',
        ],
    ),
    Page(
        slug='run',
        title='Run object',
        group='Core',
        intro=(
            '`pluto.init()` returns a run (`Op`) object. These are also exposed '
            'as the module-level `pluto.log`, `pluto.watch`, and `pluto.alert` '
            'functions, which forward to the active run.'
        ),
        symbols=[
            'op.Op.log',
            'op.Op.watch',
            'op.Op.alert',
            'op.Op.finish',
            'op.Op.add_tags',
            'op.Op.remove_tags',
        ],
    ),
    Page(
        slug='data',
        title='Structured data',
        group='Data types',
        intro='Structured data types for richer visualizations.',
        symbols=[
            'data.Data',
            'data.Graph',
            'data.Histogram',
            'data.Table',
        ],
    ),
    Page(
        slug='media',
        title='Files & media',
        group='Data types',
        intro='Log images, audio, video, text, and arbitrary artifacts.',
        symbols=[
            'file.File',
            'file.Image',
            'file.Audio',
            'file.Video',
            'file.Text',
            'file.Artifact',
        ],
    ),
    Page(
        slug='config',
        title='Settings & system',
        group='Configuration',
        intro='Client configuration and system monitoring.',
        symbols=[
            'sets.Settings',
            'sys.System',
        ],
    ),
    Page(
        slug='query',
        title='Query API',
        group='Query',
        intro=(
            'Read/query API for runs, metrics, files, and logs '
            '(`import pluto.query as pq`).'
        ),
        symbols=[
            'query.Client',
            'query.list_projects',
            'query.list_runs',
            'query.get_run',
            'query.get_metric_names',
            'query.get_metrics',
            'query.get_statistics',
            'query.compare_runs',
            'query.leaderboard',
            'query.get_files',
            'query.download_file',
            'query.get_logs',
            'query.QueryError',
        ],
    ),
]

# Public method names to render for documented classes (others are skipped).
# Falls back to "all non-underscore methods with docstrings" when not listed.
CLASS_METHODS = {
    'data.Data': ['__init__'],
    'data.Graph': ['__init__'],
    'data.Histogram': ['__init__'],
    'data.Table': ['__init__', 'add_data', 'add_column'],
    'query.Client': [
        '__init__',
        'list_projects',
        'list_runs',
        'get_run',
        'get_metric_names',
        'get_metrics',
        'get_statistics',
        'compare_runs',
        'leaderboard',
        'get_files',
        'download_file',
        'get_logs',
    ],
}


# ---------------------------------------------------------------------------
# RST -> Markdown cleanup. Docstrings mix Google style with a little RST:
#   ``code``  (double backtick)  and  literal blocks introduced by `::`.
# ---------------------------------------------------------------------------
def rst_to_md(text: str) -> str:
    if not text:
        return ''
    # RST cross-ref roles, e.g. :meth:`close` -> `close`
    text = re.sub(r':[a-zA-Z]+:`([^`]+)`', r'`\1`', text)
    # ``inline`` -> `inline`
    text = text.replace('``', '`')
    # `text::` literal block -> fenced code block.
    lines = text.splitlines()
    out: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.rstrip()
        if stripped.endswith('::'):
            head = stripped[:-2].rstrip()
            if head:  # keep the lead-in sentence, drop the trailing colon-colon
                out.append(head + ':')
            i += 1
            # skip a single blank line after the `::`
            if i < len(lines) and not lines[i].strip():
                i += 1
            block: List[str] = []
            while i < len(lines) and (
                not lines[i].strip() or lines[i].startswith('    ')
            ):
                block.append(lines[i][4:] if lines[i].startswith('    ') else lines[i])
                i += 1
            while block and not block[-1].strip():
                block.pop()
            if block:
                out.append('')
                out.append('```python')
                out.extend(block)
                out.append('```')
                out.append('')
            continue
        out.append(line)
        i += 1
    return '\n'.join(out).strip()


def esc(text: str) -> str:
    """Escape pipe chars for Markdown table cells and collapse newlines."""
    return text.replace('\n', ' ').replace('|', '\\|').strip()


def ann_str(annotation) -> str:
    if annotation is None:
        return ''
    return str(annotation)


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------
def render_signature(name: str, func, skip_self: bool = False) -> str:
    parts: List[str] = []
    star_added = False
    params = list(func.parameters)
    if skip_self and params and params[0].name in ('self', 'cls'):
        params = params[1:]
    for p in params:
        kind = p.kind.value
        if kind == 'variadic positional':
            parts.append(f'*{p.name}')
            star_added = True
            continue
        if kind == 'variadic keyword':
            parts.append(f'**{p.name}')
            continue
        if kind == 'keyword-only' and not star_added:
            parts.append('*')
            star_added = True
        piece = p.name
        ann = ann_str(p.annotation)
        if ann:
            piece += f': {ann}'
        if p.default is not None:
            piece += f' = {p.default}' if ann else f'={p.default}'
        parts.append(piece)
    ret = ann_str(func.returns)
    ret_s = f' -> {ret}' if ret else ''
    inner = ', '.join(parts)
    if len(inner) <= 76:
        return f'{name}({inner}){ret_s}'
    body = ',\n    '.join(parts)
    return f'{name}(\n    {body},\n){ret_s}'


def docstring_sections(obj):
    return obj.docstring.parsed if obj.docstring else []


def render_params_table(sections) -> str:
    for sec in sections:
        if sec.kind.value != 'parameters':
            continue
        rows = ['| Parameter | Type | Description |', '| --- | --- | --- |']
        for p in sec.value:
            ann = ann_str(p.annotation)
            type_cell = f'`{esc(ann)}`' if ann else ''
            desc_cell = esc(rst_to_md(p.description))
            rows.append(f'| `{p.name}` | {type_cell} | {desc_cell} |')
        return '\n'.join(rows)
    return ''


def render_returns(sections) -> str:
    for sec in sections:
        if sec.kind.value != 'returns':
            continue
        anns = {ann_str(r.annotation) for r in sec.value}
        # griffe splits a wrapped single-return description into multiple
        # entries that share one annotation: collapse them back into one line.
        if len(anns) == 1:
            ann = anns.pop()
            desc = rst_to_md(' '.join(r.description.strip() for r in sec.value))
            prefix = f'`{ann}` — ' if ann else ''
            return f'**Returns:** {prefix}{desc}'.rstrip()
        out = []
        for r in sec.value:
            ann = ann_str(r.annotation)
            desc = rst_to_md(r.description)
            prefix = f'`{ann}` — ' if ann else ''
            out.append(f'- {prefix}{desc}'.strip())
        body = '\n'.join(x for x in out if x)
        return f'**Returns:**\n\n{body}' if body else ''
    return ''


def render_raises(sections) -> str:
    for sec in sections:
        if sec.kind.value != 'raises':
            continue
        rows = []
        for r in sec.value:
            ann = ann_str(r.annotation)
            rows.append(f'- `{ann}` — {esc(rst_to_md(r.description))}')
        if rows:
            return '**Raises:**\n\n' + '\n'.join(rows)
    return ''


def render_example(contents: str) -> str:
    """Render an example body. `::` literal blocks are fenced by rst_to_md;
    a plain indented code body (Google `Example:` without `::`) is wrapped in
    one python fence."""
    md = rst_to_md(contents)
    if '```' in md:
        return md
    return '```python\n' + md + '\n```'


def render_admonitions(sections) -> str:
    out = []
    for sec in sections:
        if sec.kind.value not in ('admonition', 'examples'):
            continue
        if sec.kind.value == 'examples':
            # griffe yields a list of (DocstringSectionKind, str) tuples: a
            # `text` segment is prose, an `examples` segment is code.
            parts = []
            for ex_kind, content in sec.value:
                kv = getattr(ex_kind, 'value', str(ex_kind))
                if kv == 'examples':
                    parts.append('```python\n' + content.strip() + '\n```')
                else:
                    parts.append(rst_to_md(content))
            body = '\n\n'.join(p for p in parts if p)
            if body:
                out.append('**Example**\n\n' + body)
            continue
        kind = getattr(sec.value, 'kind', '')
        title = sec.title or (kind.title() if kind else 'Note')
        if kind == 'example':
            out.append(f'**{title}**\n\n' + render_example(sec.value.contents))
        else:
            out.append(f'**{title}**\n\n' + rst_to_md(sec.value.contents))
    return '\n\n'.join(out)


def lead_text(sections) -> str:
    for sec in sections:
        if sec.kind.value == 'text':
            return rst_to_md(sec.value)
    return ''


def render_function(
    obj,
    label: str,
    heading: str = '###',
    sig_name: Optional[str] = None,
    skip_self: bool = False,
) -> str:
    sections = docstring_sections(obj)
    blocks = [f'{heading} `{label}`', '']
    blocks.append('```python')
    blocks.append(render_signature(sig_name or label, obj, skip_self=skip_self))
    blocks.append('```')
    for chunk in (
        lead_text(sections),
        render_params_table(sections),
        render_returns(sections),
        render_raises(sections),
        render_admonitions(sections),
    ):
        if chunk:
            blocks.append('')
            blocks.append(chunk)
    return '\n'.join(blocks)


def render_class(obj, symbol_key: str) -> str:
    sections = docstring_sections(obj)
    blocks = [f'### `{obj.name}`', '']
    lead = lead_text(sections)
    if lead:
        blocks.append(lead)
    # A class docstring may itself carry an Args/Attributes section.
    for chunk in (render_params_table(sections), render_admonitions(sections)):
        if chunk:
            blocks.append('')
            blocks.append(chunk)
    method_names = CLASS_METHODS.get(symbol_key)
    if method_names is None:
        # Always document the constructor; otherwise public, documented methods.
        documented = [
            n
            for n, m in obj.members.items()
            if m.kind.value == 'function'
            and not n.startswith('_')
            and m.docstring is not None
        ]
        method_names = (['__init__'] if '__init__' in obj.members else []) + documented
    for mname in method_names:
        if mname not in obj.members:
            continue
        method = obj.members[mname]
        if method.kind.value != 'function':
            continue
        if mname == '__init__':
            label, sig_name = f'{obj.name}(...)', obj.name
        else:
            label, sig_name = f'{obj.name}.{mname}', mname
        blocks.append('')
        blocks.append(
            render_function(
                method, label, heading='####', sig_name=sig_name, skip_self=True
            )
        )
    return '\n'.join(blocks)


def render_symbol(pkg, symbol_key: str) -> str:
    obj = pkg[symbol_key]
    if obj.kind.value == 'class':
        return render_class(obj, symbol_key)
    # function / method. Methods (parent is a class) are bound on the run/client
    # object when called, so drop the leading `self`.
    is_method = getattr(obj.parent, 'kind', None) and obj.parent.kind.value == 'class'
    return render_function(obj, obj.name, skip_self=bool(is_method))


def render_page(pkg, page: Page) -> str:
    fm = [
        '---',
        f'title: {json.dumps(page.title)}',
        '---',
        '',
        '{/* AUTO-GENERATED by scripts/gen_api_docs.py — do not edit by hand. */}',
        '',
    ]
    if page.intro:
        fm.append(page.intro)
        fm.append('')
    body = [render_symbol(pkg, s) for s in page.symbols]
    return '\n'.join(fm) + '\n' + '\n\n---\n\n'.join(body) + '\n'


def build() -> dict:
    pkg = griffe.load('pluto', search_paths=[str(REPO_ROOT)], docstring_parser='google')
    files = {}
    meta_pages = []
    for page in PAGES:
        files[f'{page.slug}.mdx'] = render_page(pkg, page)
        meta_pages.append(
            {
                'slug': page.slug,
                'file': f'{page.slug}.mdx',
                'title': page.title,
                'group': page.group,
                'symbols': [f'pluto.{s}' for s in page.symbols],
            }
        )
    # Resolve the documented public surface from pluto.__all__ for traceability.
    try:
        public = list(pkg.exports or [])
    except Exception:
        public = []
    meta = {
        'version': META_VERSION,
        'package': 'pluto',
        'package_version': str(pkg.members.get('__version__').value).strip('\'"')
        if '__version__' in pkg.members
        else None,
        'public_api': sorted(str(x) for x in public),
        'pages': meta_pages,
    }
    files['meta.json'] = json.dumps(meta, indent=2) + '\n'
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        '--check',
        action='store_true',
        help='Exit non-zero if committed output is stale (for CI).',
    )
    ap.add_argument('--out', default=str(OUT_DIR), help='Output directory.')
    args = ap.parse_args()
    out = Path(args.out)
    files = build()

    # Files the generator owns (everything else, e.g. README.md, is left alone).
    # Used to detect/prune orphans from pages that were removed from PAGES.
    def managed_on_disk() -> set:
        if not out.exists():
            return set()
        found = {p.name for p in out.glob('*.mdx')}
        if (out / 'meta.json').exists():
            found.add('meta.json')
        return found

    if args.check:
        stale = []
        for name, content in files.items():
            path = out / name
            if not path.exists() or path.read_text() != content:
                stale.append(name)
        orphans = sorted(managed_on_disk() - set(files))
        if stale or orphans:
            print('Stale API docs (run scripts/gen_api_docs.py):', file=sys.stderr)
            for name in stale:
                print(f'  - changed/missing: {name}', file=sys.stderr)
            for name in orphans:
                print(f'  - orphaned (remove): {name}', file=sys.stderr)
            return 1
        print(f'docs-api/ is up to date ({len(files)} files).')
        return 0

    out.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (out / name).write_text(content)
    # Prune managed files left over from pages removed from PAGES.
    removed = sorted(managed_on_disk() - set(files))
    for name in removed:
        (out / name).unlink()
    msg = f'Wrote {len(files)} files to {out}/'
    if removed:
        msg += f' (pruned {len(removed)}: {", ".join(removed)})'
    print(msg)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
