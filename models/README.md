# Frozen model weights

The three state dictionaries correspond to seeds 17, 42, and 73 of the frozen
January--September training protocol. The files contain model parameters only;
optimizer states and raw waveforms are not included.

The public loader converts archived checkpoints to the mathematically
equivalent effective inference graph; equivalence is covered by tests. Detailed
compatibility provenance is retained in `artifacts/summary/ARTIFACT_SCOPE.md`.

Institutional approval for public redistribution of trained weights must be
confirmed before this directory is uploaded. If weights cannot be released,
publish the architecture and configuration, retain the synthetic random-weight
smoke test, and explain the restriction in the software and data availability
statements.
