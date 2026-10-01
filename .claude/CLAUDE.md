# tokimeki — Project Guide

Finds the cute moments of anime heroines across a series and cuts them into MADs. Stage 1 is a scene library, stage 2 the music-synced cut; `README.md` holds the design and is kept current as decisions are made.

## Hard Rules

- **Strict types, zero tolerance.** `pixi run check` (ruff lint, ruff format check, basedpyright strict) passes on every commit. No `# type: ignore` or `# pyright: ignore` without a stated reason; untyped third-party APIs get a typed wrapper in one place instead of `Any` leaking through.
- **The content filter comes first.** Shots WD14 rates questionable or explicit are dropped before any other stage sees them: they never enter the library and are never sent to a cloud model.
- **Media never goes into git.** Episodes, songs, subtitles, frames, databases and renders live in the data directory outside the repository.
- **Nothing downloads sources.** The user provides the episodes; no BitTorrent or scraping in this code or in sessions.
- **No bursts of cloud calls or agent sessions.** Batch API work is rate-limited and resumable; never start many Claude/Codex sessions in a short time (account ban risk).
- **One GPU model at a time.** The GPU has 8 GB; stages load a model, run it over the batch, and free it.

## Working Style

- Reply to the user in 简体中文; code, comments, commits and docs in English.
- Every pipeline stage writes its results to the series database or the cache, so a rerun skips finished work and one stage can be redone alone.
- Each stage ships with pytest tests on small fixtures (a few frames, a short clip, a few subtitle lines), never on copyrighted episodes.
- Zero comments by default; one only for a non-obvious why.
- Long jobs run in the background with a completion notice; no foreground sleep/poll loops.

## Build, Check, Test

- `pixi install` — environment; dependencies are added with the code that first needs them.
- `pixi run check` — ruff + basedpyright strict. `pixi run test` — pytest.
- CI (`.github/workflows/ci.yml`) runs `pixi run check` on every push and PR.

## Commits

Conventional style, lowercase, imperative: `feat: …`, `fix: …`, `test: …`, `chore: …`, `docs: …`. One logical change per commit. Commit only when asked; push only when asked.
