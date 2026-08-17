# Repository working rules

## Mandatory research log

Every change to this repository must update
`docs/research_and_development_log.md` in the same work unit. This applies to:

- algorithms, equations and verification semantics;
- vLLM hooks, workers, model/checkpoint adapters and serving scripts;
- benchmark protocols, datasets, evaluators and report generators;
- default parameters, environment variables and runtime assumptions;
- tests, bug fixes, failed experiments and decisions to stop a direction;
- branches, checkpoints or results that become invalid or superseded.

Each new log entry must include the date, status, what changed, affected files,
validation performed, and any remaining limitation. Never claim that a GPU
smoke test or benchmark passed unless it actually ran. Experimental result
artifacts remain local under ignored `results/` and `logs/` directories unless
the user explicitly asks to publish them.

When an older document becomes inaccurate, add a visible historical/superseded
notice instead of silently leaving contradictory instructions.
