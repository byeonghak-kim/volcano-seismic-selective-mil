# Data availability

The raw waveform archive and the operational catalogue are not redistributed in
this repository. Users must obtain any third-party waveform data through the
relevant provider and comply with the provider's terms. The final article must
state the precise access route and any applicable restrictions.

To preserve computational reproducibility when raw records cannot be bundled,
this release includes:

- the complete frozen model and inference implementation;
- portable keyed-manifest archive construction, preprocessing, overlap grouping,
  and temporal/group split code;
- W2 MIL, both 120-s CNN recipes, and four 144-feature baseline fitting paths;
- validation-only calibration, probability-adjustment, threshold-search,
  risk-coverage, matched-count, bootstrap, and random-referral code;
- three trained state dictionaries, subject to institutional release approval;
- normalization, calibration, and referral-policy parameters;
- aggregate result and bootstrap tables;
- deterministic synthetic inference and archive-to-bootstrap research examples;
  and
- tests for preprocessing, archive keys, overlap groups, temporal splitting,
  feature dimensions, tensor contracts, policies, resampling, and inference.

The synthetic signals are not substitutes for the study data and must not be
used to estimate scientific performance.
