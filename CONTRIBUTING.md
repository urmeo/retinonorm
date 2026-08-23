# Contributing to CortexProbe

Thanks for your interest. CortexProbe is a measurement instrument: it fits population
receptive fields to convolutional unit activations and is validated against synthetic ground
truth. Contributions that strengthen the validation, widen the failure modes it catches, or
make the numbers easier to reproduce are the most useful.

## Development setup

```bash
python3 -m pip install -e '.[dev]'
```

Optional extras: `.[models]` pulls PyTorch and torchvision for probing real networks,
`.[viz]` pulls matplotlib, `.[io]` pulls pandas and pyarrow.

## Before opening a pull request

CI runs on Python 3.9 and 3.12. Run the same four checks locally; all should be clean:

```bash
python -m ruff format --check .
python -m ruff check .
python -m mypy
python -m pytest -q --cov
```

CI also re-derives the validation report and fails if any number moved:

```bash
python scripts/generate_validation_report.py --check
```

If a change is meant to move a number, regenerate the report in the same commit and say in
the description which number moved and why.

## Scope

The instrument is validated; no network has been probed yet. Pull requests that report
results from a probed network should include the fit diagnostics and the rejection counts,
not only the headline fit.
