# `docs-api/` — generated Python API reference (Mintlify)

**Auto-generated. Do not edit these `.mdx` files by hand** — they are
overwritten by `scripts/gen_api_docs.py`. Edit the Python docstrings instead,
then regenerate.

## What this is

MDX documentation for Pluto's public Python API, generated statically from
docstrings with [griffe](https://mkdocstrings.github.io/griffe/). It is the
*producer* side of a pull-based docs pipeline: the Mintlify docs repo
(`Trainy-ai/konduktor`) consumes this directory and builds its own navigation.

## Regenerate

```bash
pip install griffe              # or: poetry install --with dev
python scripts/gen_api_docs.py  # writes docs-api/*.mdx + meta.json
```

CI (`.github/workflows/api-docs.yml`) runs `--check` on every PR and fails if
the committed output is stale.

## What gets documented

The page set lives in the `PAGES` manifest in `scripts/gen_api_docs.py`,
anchored to `pluto.__all__`. To document a new public symbol: add it to
`pluto/__init__.py:__all__`, add it to a page in `PAGES`, and regenerate.

## Consuming from the Mintlify repo

This directory is a stable contract. For each tag/commit you pin, expect:

- One `<slug>.mdx` file per page (filenames are stable).
- `meta.json` — the navigation manifest:

  ```jsonc
  {
    "version": 1,                 // schema version; guard against bumps
    "package": "pluto",
    "package_version": "0.0.23",
    "public_api": [...],          // names from pluto.__all__
    "pages": [
      {
        "slug": "initialization",
        "file": "initialization.mdx",
        "title": "Initialization & lifecycle",
        "group": "Core",          // suggested nav group (regroup freely)
        "symbols": ["pluto.init.init", ...]
      }
    ]
  }
  ```

A typical consumer CI step checks out this repo at a pinned tag, copies
`docs-api/*.mdx` into the Mintlify tree, and builds `docs.json` navigation from
`meta.json` (`pages[].group` → group, `pages[].file` → page).
