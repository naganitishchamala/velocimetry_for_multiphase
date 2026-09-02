# Contributing

## Development setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
pytest
```

## Before opening a pull request

- `pytest` passes.
- New numerical behaviour has a test. For anything touching uncertainty, prefer
  a test that measures the quantity empirically (generate noise realisations,
  observe the scatter) over one that restates the formula — a test written from
  the formula cannot catch a wrong formula. See `tests/test_statistics.py`.
- Changes to the HDF5 layout are noted in `CHANGELOG.md`. Downstream plotting
  scripts depend on that schema.

## Things worth knowing

**The uncertainty datasets mean specific things.** `residual` is a *signed*
per-frame displacement error following `flow_corrected = flow - residual`.
`uncertainty` is its magnitude. Neither is a standard deviation: part of the
residual is systematic and does not average away. `postprocess.velocity` splits
the two before propagating them, and that split is why the reported uncertainty
of a mean does not collapse to zero over a long record.

**`valid_fraction` is not optional.** The estimator returns numbers everywhere,
including where it has no information — decorrelated regions, residuals larger
than the search window, reflection-padded borders. Statistics that ignore the
mask will silently average in those points.

**Block coordinates come from `pipeline_utils.io.block_centers`.** Do not
assume `index * window + window // 2`; for even windows that is off by half a
window, which becomes a spatial registration error in every derived profile.
