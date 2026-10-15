# Summary artifact scope

The files in this directory support the v0.7 manuscript result and are versioned
by the frozen analysis that produced them.

- `primary_policy_metrics_v36.csv`: validation and October--December metrics for
  the final W2 MIL class-A probability-adjustment policy.
- `primary_policy_parameters_v36.csv`: validation-selected multiplier context,
  class-specific thresholds, and realized validation coverage for that policy.
- `random_referral_controls_v36.csv`: 10,000-draw controls at 398 accepted events.
- `class_A_adjustment_ablation_v39.csv`: unadjusted and adjusted W2
  probabilities, with the same validation performance-constraint search separately
  refitted at each probability stage; this is the source for Supplementary
  Table S10.
- `class_A_adjustment_ablation_protocol_v39.json`: source paths, multiplier,
  validation performance constraints, threshold-search function, and split
  provenance for S10.
- `selective_model_comparison_v37.csv`: validation-locked and equal-count
  comparisons across seven models.
- `validation_locked_policies_v37.csv`: thresholds selected without temporal-test
  labels for the seven-model comparison.
- `risk_coverage_curves_v37.csv`: sampled temporal-test risk-coverage points;
  AURC and AUGRC in the comparison table were computed from all ranked events.
- `selective_comparison_protocol_v37.json`: comparison protocol metadata.
- `primary_group_bootstrap_ci_v38.csv`: 10,000-repetition overlap-group bootstrap
  intervals for the final primary operating point.
- `primary_bootstrap_protocol_v38.json`: bootstrap protocol metadata.
- `public_release_prediction_verification.json`: public inference versus frozen
  research predictions for all 574 chronological-test events.
- `model_only_runtime.csv` and `runtime_environment.json`: architecture-level
  runtime measurements; these are performance-engineering records, not
  classification-result tables. The W2 timing was recorded with the v35
  compatibility graph (836,491 stored parameters). The public loader removes
  two zero-input columns and inactive auxiliary heads, leaving 834,182 effective
  inference parameters without changing logits. The measured latency is retained
  as a conservative frozen benchmark and the two counts are reported separately.

The superseded v30 primary/comparison CSVs are retained in the internal research
outputs but intentionally omitted here, because they describe the model and
policy before the v36 class-A probability adjustment and would conflict with the
v0.7 manuscript endpoint.
