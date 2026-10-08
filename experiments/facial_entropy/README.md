# experiments/facial_entropy

- `protocol.md` — the frozen protocol (rules implemented by `src/facial_entropy/`).
- `configs/{smoke,quick,main}.toml` — profiles (`smoke` = synthetic only, offline; `quick` = 50 FEI subjects; `main` = all 200 FEI subjects, 500 requested).
- `reports/source_audit.md` — audit of the reused Hyperplane-Forge checkout (paths, symbols, defaults, commit, discrepancies).

Run outputs live under `artifacts/facial-entropy/<profile>/` (see the project README for the exact commands); each contains `results.md` as the readable report.
