---
name: model-diagram-reconciliation
description: Create auditable LambdaRank and LambdaMART editorial diagrams
version: 1.0.0
metadata:
  hermes:
    tags: [pytorch, lightgbm, diagrams, model-card]
    category: AI
---

# Model diagram reconciliation

## When to use
Use when producing the two senior-audience model-flow SVGs.

## Procedure
1. Locate the selected stable trial and exact packaged run, if available.
2. Read implementation and configuration, not only a prior report.
3. Draw LambdaRank: grouped date list → feature input → configured normalization/hidden layers → scalar score → within-date pairs → NDCG-weighted LambdaRank objective → tail mapping.
4. Draw LambdaMART: grouped date list → feature matrix → shallow boosted trees → rank score → NDCG/LambdaRank objective → tail mapping.
5. Include effective width/depth, activation, normalization, dropout, backend, leaves/depth, truncation, and label semantics when present.
6. Add text alternatives and keep diagrams readable at 768px.

## Guardrails
Do not use Torchview. Do not draw every tensor operation or every tree. Do not invent parameters when the selected artifact is missing; label the gap.
