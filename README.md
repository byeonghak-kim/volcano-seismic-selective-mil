# Selective MIL for volcano-seismic waveform archives

This repository is the public-release candidate accompanying a study of JMA
type-A volcanic earthquakes, type-B volcanic earthquakes, and volcanic tremor.
It implements the frozen final system and a portable, method-equivalent
reproduction path for the main chronological-holdout analysis.

The final model is a **two-record waveform MIL ensemble (W2 MIL)**. Each
event is represented by two consecutive 60-s, three-component waveform records.
Both records are retained. A soft record-informativeness proxy biases gated
attention but never deletes a record. Three independently initialized models
are temperature-scaled on validation data, averaged at the probability level,
followed by a validation-selected class-A probability adjustment
(`m_A = 0.2546415841`) and renormalization,
and then passed through validation-fixed predicted-class-specific accept/refer
thresholds.

W2 denotes the two consecutive 60-s records retained as separate MIL
instances; it is not a new neural-network layer or an asserted 120-s event
duration.

The public loader converts archived checkpoints to the mathematically
equivalent effective inference graph; equivalence is covered by tests.

## Main chronological result

Partitioning used catalogue event time, defined by the first record. Training
and validation catalogue rows have first-record times in January--September
2023, whereas temporal-test rows have first-record times in October--December.
The prescribed follow-on record remained with its event even when it crossed a
calendar boundary; no physical record or overlap component crossed partitions.
October--December temporal-test catalogue rows were excluded from all
parameter-fitting calculations, including normalization, supervised fitting,
checkpoint selection, temperature fitting, and threshold selection.

| Model | Realized coverage | Accepted accuracy | Accepted macro-F1 | Error capture |
|---|---:|---:|---:|---:|
| W2 MIL (class-A probability adjustment) | 69.34% | 96.48% | 92.46% | 78.79% |
| 120-s concatenation CNN (standard training) | 47.56% | 94.14% | 93.73% | 84.91% |
| 120-s concatenation CNN (imbalance-adjusted training) | 77.53% | 92.58% | 86.47% | 64.13% |

The W2 row is the frozen v36 primary operating point. The CNN rows use the
v37 `validation-locked class-aware @70%` policy reported in
`artifacts/summary/selective_model_comparison_v37.csv`.

These are retrospective internal temporal-validation results, not prospective
cross-observatory validation.

## Model-only inference cost

Measured in FP32 on the development workstation with inputs already normalized
and resident on the target device:

| System | Device | Batch | Median batch latency | Throughput |
|---|---|---:|---:|---:|
| W2 soft-q MIL, single model | CPU, 8 threads | 1 | 42.5 ms | 23.5 events/s |
| W2 soft-q MIL, 3-seed ensemble | CPU, 8 threads | 1 | 119.2 ms | 8.4 events/s |
| W2 soft-q MIL, single model | RTX 3080 Ti | 1 | 3.36 ms | 297 events/s |
| W2 soft-q MIL, 3-seed ensemble | RTX 3080 Ti | 1 | 10.07 ms | 99 events/s |
| W2 soft-q MIL, 3-seed ensemble | RTX 3080 Ti | 32 | 34.33 ms | 932 events/s |

These measurements exclude waveform file I/O, filtering, resampling, and
quality-score computation. Full results and environment metadata are supplied
in `artifacts/summary/model_only_runtime.csv` and
`artifacts/summary/runtime_environment.json`.

## Repository layout

- `src/ve_selective/`: model, quality score, grouping, policy, and inference code.
- `src/ve_selective/preprocessing.py` and `splitting.py`: keyed archive construction,
  waveform preprocessing, overlap groups, and temporal/group splitting.
- `src/ve_selective/training.py`: W2 MIL and both 120-s CNN fitting recipes,
  validation checkpoint selection, and temperature calibration.
- `src/ve_selective/features.py` and `baselines.py`: the 72-feature-per-record,
  144-feature-per-event representation and LR/RF/RBF-SVM/HGB searches.
- `src/ve_selective/policy_search.py`, `evaluation.py`, and `pipeline.py`:
  probability adjustment, validation-locked thresholds, risk-coverage,
  matched-count diagnostics, bootstrap, and random-referral controls.
- `configs/final_w2_temporal.json`: frozen normalization, class-prior,
  calibration, and referral parameters.
- `configs/research_pipeline.json`: method-equivalent sanitized research fitting
  and evaluation configuration; paths and raw data are deliberately not embedded.
- `schemas/`: portable event and physical-record manifest interfaces.
- `models/`: three frozen state dictionaries.
- `scripts/synthetic_demo.py`: model-and-policy inference on synthetic preprocessed tensors.
- `scripts/synthetic_research_pipeline.py`: raw-data-free archive-to-bootstrap smoke run.
- `scripts/build_keyed_archive.py`, `build_temporal_split.py`, and
  `train_research_pipeline.py`: command-line research reproduction path.
- `scripts/reproduce_summary_tables.py`: exports the supplied, versioned aggregate tables.
- `artifacts/summary/`: v36 primary-policy metrics and parameters, v37
  validation-locked seven-model comparisons and risk-coverage curves, v38
  primary group-bootstrap intervals, the final 574-event software-verification
  record, and runtime benchmarks. No event waveforms are included.
- `tests/`: model, grouping, and selective-policy smoke tests.

## Installation

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

GPU execution requires a PyTorch build compatible with the local CUDA driver.
CPU-only execution is supported.

The exact core package versions used for the final public-package verification
are recorded in `environment/verified_environment.json` and
`environment/requirements-verified.txt`. The ordinary requirement files retain
compatible lower bounds; the verified files provide the concise reproducibility
snapshot referenced by the article.

## Research-code reproduction scope

The research code is independent of the original institutional folder layout.
It uses two keyed CSV files described in `schemas/README.md`:

- one event row with `event_id`, A/B/T label, first-record catalogue time, and
  two physical station-minute record keys; and
- one record row mapping each key to a relative `.npy` array and its original
  sampling rate.

The `.npy` record has shape `(3, samples)` in E, N, U/Z order and contains
finite stored/native SAC sample amplitudes. Absolute paths and paths escaping
the supplied record root are rejected. Raw waveform and catalogue data are not
included in this repository.

The fitting code is a method-equivalent sanitized reimplementation, not a
bitwise replay of the archived training executable. The supplied frozen W2
weights define the manuscript endpoint. Re-training preserves the documented
data flow, architecture, objectives, sampling, validation selection,
calibration, and policy logic, while implementation-level initialization,
precision, and execution details can produce different fitted weights; use the
pipeline for method reproduction and sensitivity analysis, not exact checkpoint
checksum recreation.

The full portable path is:

```bash
python scripts/build_keyed_archive.py \
  --events local_data/events_manifest.csv \
  --records local_data/records_manifest.csv \
  --records-root local_data/records \
  --output research_output/archive

python scripts/build_temporal_split.py \
  --events research_output/archive/events.csv \
  --cutoff 2023-10-01T00:00:00 \
  --output research_output/split_manifest.csv

python scripts/train_research_pipeline.py \
  --archive research_output/archive \
  --split-manifest research_output/split_manifest.csv \
  --config configs/research_pipeline.json \
  --output research_output/results
```

The training pipeline fits the three W2 seeds and both three-seed CNN families,
calibrates each neural seed on validation rows, extracts and fits the four
144-feature baselines, and applies the same validation-locked global and
class-specific threshold searches to all seven model families. It also writes
complete risk-coverage curves with AURC/AUGRC, the label-free matched-count
diagnostic, overlap-group bootstrap intervals, and random-referral controls.
Test labels are used for final metric computation, never for model,
temperature, adjustment, or threshold fitting.

For a short executable fixture covering archive construction, preprocessing,
splitting, W2/CNN/classical fitting, calibration, policy search, evaluation,
and resampling without proprietary data, run:

```bash
python scripts/synthetic_research_pipeline.py \
  --output demo_output/research_pipeline \
  --config configs/research_pipeline.json
```

The fixture uses one epoch, one seed, one training batch per neural model, and
20 resamples; it verifies software paths and is not scientific evidence. The
frozen manuscript-facing CSV/JSON files remain preserved under
`artifacts/summary/`. Exact manuscript values require the unavailable original
waveforms and catalogue rows.

## Synthetic preprocessed-input demo

```bash
python scripts/synthetic_demo.py --device cpu --output demo_output/predictions.csv
```

The demo constructs deterministic, synthetic three-component signals, computes
window-informativeness scores, loads all three frozen models, applies training-
only normalization, log-prior adjustment, validation-only temperatures,
probability averaging, the validation-selected class-A probability adjustment, and
frozen class-specific accept/refer thresholds. The
synthetic predictions are a software test and are not scientific performance
evidence.

The release implementation was checked against all 574 chronological-test
events available in the local research environment. The bundled event-level
verification record covers the corrected and renormalized final
69.3379%-coverage policy.

`artifacts/summary/public_release_prediction_verification.json` records hashes
of the private 574-event input tensor, private-derived final probability
reference, frozen configuration, and three supplied checkpoints together with
the verification runtime and device; the private event-level artifacts
themselves are not distributed. An authorized holder can repeat the check with
a float32 `(N, 2, 3, 6000)` preprocessed `.npy` tensor and a row-aligned CSV of
final post-adjustment `p_A`, `p_B`, and `p_T` values:

```bash
python scripts/verify_private_predictions.py \
  --input PRIVATE/test_waveforms.npy \
  --reference PRIVATE/final_reference_probabilities.csv \
  --device cpu \
  --output private_prediction_verification.json
```

## Tests

```bash
pytest -q
```

Without the optional `pytest` dependency, the same core assertions can be run
directly:

```bash
python tests/test_smoke.py
```

## Reproducing aggregate tables

```bash
python scripts/reproduce_summary_tables.py --output demo_output
```

This exports the current manuscript-facing v36 primary-policy, v37
selective-comparison, and v38 bootstrap tables. Earlier v30 aggregate comparison
CSVs are deliberately excluded from this package because they describe the
superseded pre-adjustment model and policy, not the v0.7 manuscript result.
Event-level metrics requiring the original waveforms and catalogue labels cannot
be recomputed from this public bundle alone; see `DATA_AVAILABILITY.md`.

## Using preprocessed waveform tensors

`SelectiveEnsemble.predict` is not a raw-waveform preprocessing entry point. It
accepts only finite, real-valued tensors shaped `(N, 2, 3, 6000)` at 100 Hz.
The axes are event, record, component, and sample; component order is E, N,
U/Z. The tensors are before the frozen training-set normalization, which
`predict` applies internally.

SAC sample amplitudes were used in stored/native units; no instrument-response
correction or cross-sensor unit conversion was applied. The frozen
normalization statistics are valid only for this stored/native-amplitude input
domain.

Prepare each physical 60-s station-minute record component by component using
the same upstream recipe as the study:

1. Convert the component to `float64` and subtract its sample mean.
2. Design a SciPy order-4, 0.5--30 Hz Butterworth band-pass in second-order
   sections at the record's original sampling rate. Apply `sosfilt` forward,
   reverse that output, apply `sosfilt` again, and reverse back.
3. If needed, Fourier-resample with `scipy.signal.resample` to
   `round(n_samples * 100 / original_fs)` samples.
4. Cast to `float32`; retain samples from index 0 through 5999 when longer than
   6000, or append zeros on the right when shorter.
5. Stack components as E, N, U/Z and two consecutive records as the two bag
   instances.

The following component-level example mirrors those operations:

SciPy is required for this upstream example, but it is not a model-inference
dependency once the preprocessed tensors have been prepared.

```python
import numpy as np
from scipy.signal import iirfilter, resample, sosfilt, zpk2sos

def preprocess_component(raw, original_fs):
    x = np.asarray(raw, dtype=np.float64)
    x = x - np.mean(x)
    nyquist = 0.5 * float(original_fs)
    z, p, k = iirfilter(
        4,
        [0.5 / nyquist, 30.0 / nyquist],
        btype="band",
        ftype="butter",
        output="zpk",
    )
    sos = zpk2sos(z, p, k)
    x = sosfilt(sos, x)
    x = sosfilt(sos, x[::-1])[::-1]
    if not np.isclose(original_fs, 100.0):
        x = resample(x, int(round(len(x) * 100.0 / original_fs)))
    x = np.asarray(x, dtype=np.float32)
    if len(x) >= 6000:
        return x[:6000]
    return np.pad(x, (0, 6000 - len(x)))
```

Do not concatenate or split events across train/validation/test after grouping:
bags sharing a physical station-minute record must remain in one partition.

```python
from pathlib import Path
import numpy as np
from ve_selective.inference import SelectiveEnsemble

waveforms = np.load("local_data/events.npy")
system = SelectiveEnsemble.from_release(Path("."), device="cpu")
result = system.predict(
    waveforms,
    sampling_rate_hz=100.0,
    input_domain="preprocessed_waveform_tensor",
)
print(result[["prediction", "confidence", "decision"]])
```

## Methodological safeguards

1. Events whose bags share the same physical station-minute record are connected into overlap groups.
2. A group never crosses training, validation, or test partitions.
3. Groups containing more than one catalogue class are excluded.
4. Normalization is fitted on training waveforms only.
5. Checkpoints are selected by full-coverage validation macro-F1.
6. Temperature and referral thresholds are fitted on validation predictions only.
7. Test coverage is reported as realized; it is never retuned to the nominal target.

## Citation and release metadata

### Authors and contact

| Author | Email |
|---|---|
| Byeonghak Kim | [bhkim715@korea.kr](mailto:bhkim715@korea.kr) |
| Kyungmin Min | [mkmzx@korea.kr](mailto:mkmzx@korea.kr) |
| Eunyoung Jo (corresponding author) | [jey0817@korea.kr](mailto:jey0817@korea.kr) |
| Eui-Hong Hwang | [hkawon1@korea.kr](mailto:hkawon1@korea.kr) |

All authors: Earthquake and Volcano Research Division, Korea Meteorological
Administration, Seoul 07062, Republic of Korea.

Software authors are recorded in `CITATION.cff`. Repository URL, release date,
and archival DOI will be added when they exist. Article metadata will be added
after publication; the software title does not imply a published article.

The authors confirmed institutional authorization for public release of code
and trained weights under the MIT license on 15 October 2026.
See `RELEASE_CHECKLIST.md` for remaining release steps.
