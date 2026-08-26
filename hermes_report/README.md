# Hermes post-study report bundle

This package adds four local Hermes skills and a bundle definition for
producing one evidence-backed HTML story from the completed Optuna, MLflow,
and packaged-run artifacts.

## Install order

From the repository root:

```bash
mkdir -p .hermes/skills
cp -R hermes_report/skills/* .hermes/skills/
hermes skills trust .
hermes skills list
hermes bundles create l2r-post-study-report \
  --skill ml-evidence-audit \
  --skill ranking-storytelling \
  --skill model-diagram-reconciliation \
  --skill portable-html-report \
  -d "Build the Learning2Rank post-study report"
```

Hermes does not install local directories through the hub command. The commands
above register the repo-local skills. If the profile cannot load repo-local
skills, copy the four directories into `~/.hermes/skills/` instead.

## Run

```bash
hermes chat --toolsets skills -q \
  "/l2r-post-study-report Read DIRECTORS_SCRIPT.md and build the report."
```

The output contract is defined in `DIRECTORS_SCRIPT.md`. Perspective is
intentionally excluded. The report must be self-contained and work offline
after build; charts may use inline SVG or a vendored Canvas-compatible runtime.
