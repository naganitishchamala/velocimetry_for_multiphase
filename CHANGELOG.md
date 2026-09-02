# Changelog

## 0.2.0 — unreleased

### Corrected

- **Reynolds-stress uncertainty was inflated by √N.** An uncommitted local edit
  had replaced the `1/N²` prefactor in the delta-method propagation with `1/N`,
  overstating σ⟨u′u′⟩ and σ⟨u′v′⟩ by ~116× on a 13,000-frame record. Restored,
  and now guarded by tests that compare against measured scatter.
- **Uncertainty of the mean no longer ignores sample size.** It was the RMS of
  the instantaneous σ, independent of record length. It is now
  `sqrt(bias² + var_random / N_eff)`, with `N_eff = N / (2·τ_int)` from the
  measured integral timescale. The instantaneous precision is still reported,
  as its own quantity.
- **Reynolds stresses are corrected for noise inflation.** Random measurement
  error adds σ² to a measured variance; `reynolds_stresses_noise_corrected`
  removes it. The uncorrected values are kept alongside.
- **Spatial registration was off by half a window.** Block averaging used a
  padded `filter2D` with a strided slice, sampling block *k* at `k·w − 0.5`
  while consumers placed it at `k·w + w//2` — a 2.5 px offset at `w = 4`, and a
  corrupted first block. Now exact non-overlapping blocks, with
  `block_centers()` as the single source of truth for coordinates.
- **Shedding frequency assumed 130 kHz.** `compute_shedding_frequency` now
  requires an explicit rate and the CLI reads `fps_capture` from the file.
  Segment length raised (127 Hz → 16 Hz bins) and the peak search band-limited.
- **`--angle` silently did nothing** while reporting that it had rotated the
  field. It now raises.
- **Cavity statistics divided by the container's frame-count estimate** rather
  than frames actually read. Now single-pass Welford over the real count.
- **OpenPIV filled gaps outside the interrogation hull with zero velocity**,
  which entered the statistics as data. Now NaN, excluded via the validity mask,
  and OpenPIV's own outlier flags are honoured instead of discarded.
- **12-bit recordings were quantised to 8 bits** in the first preprocessing
  step, discarding the dynamic range that motivates using `.cine` at all. Now
  scaled in float, with the bit depth inferred or set by `--bit_depth`.
- Frame pairs are read sequentially instead of by seeking per pair, which is
  unreliable on inter-frame-compressed containers and could pair wrong frames.
- Statistics accumulate in float64 throughout.

### Added

- `residual` (signed) and `valid_fraction` datasets; points where the estimator
  has no information are flagged rather than reported as confident.
- Configurable uncertainty search half-width (`--uncert_search`). Residuals
  beyond it are marked invalid instead of clamped to a smaller value.
- RAFT's trailing GRU iterates are returned; their spread is a convergence-based
  uncertainty available at no extra cost.
- Full provenance in every HDF5 file — calibration, method, model, CLAHE
  settings, frame count, code version — mirrored onto each dataset and carried
  into derived files.
- `integral_timescale()`, and `n_effective` in extracted profiles.
- Test suite (31 tests) covering preprocessing, block registration, the
  uncertainty estimator's contract, propagation against measured scatter, and
  an end-to-end tracker run.
- Packaging: `pyproject.toml`, `requirements.txt`, `LICENSE`, `NOTICE`.

### Changed

- `--imgScale` and `--fpsCam` are now required. Their previous defaults
  (0.001 mm/px, 130000 Hz) were specific to one laboratory and silently wrong
  for anyone else.
- Publication figure code moved out of the repository; the tool writes
  self-describing HDF5 and plotting scripts read it.
- The uncertainty estimator is ~3.6× faster: frame-1 statistics are hoisted out
  of the shift loop and the box filter is applied separably.
- `predict_batch` returns an `UncertaintyResult` rather than a bare array.

### Known limitations

- The uncertainty has not been validated against ground truth. A coverage test
  on synthetic pairs with known flow is the next step; until then σ should be
  read as an indicator, not a calibrated interval.
- CLAHE is applied per frame, which weakens brightness constancy. Compare
  results with and without it.
- The uncertainty budget covers image matching only — not calibration, timing,
  out-of-plane motion, or the tracker's own smoothness prior.
