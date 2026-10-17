"""Benchmark input errors are clear before any fitting begins."""

from __future__ import annotations

import pytest

from .conftest import load_script


@pytest.fixture
def benchmark():
    return load_script("benchmarks/benchmark_scaling.py")


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_make_units_rejects_invalid_counts(benchmark, grid, apertures, count):
    with pytest.raises(ValueError, match="positive integer"):
        benchmark.make_units(grid, apertures, count)


@pytest.mark.parametrize("argument", ["0", "-1", "1.5", "many"])
def test_benchmark_rejects_invalid_max_units(benchmark, argument, capsys):
    with pytest.raises(SystemExit) as error:
        benchmark.main(["--max-units", argument])
    assert error.value.code == 2
    assert "positive integer" in capsys.readouterr().err


@pytest.mark.parametrize("resolution", ["0", "-2", "8", "60", "63", "large"])
def test_benchmark_rejects_unsupported_resolution(benchmark, resolution, capsys):
    with pytest.raises(SystemExit) as error:
        benchmark.main(["--resolution", resolution])
    assert error.value.code == 2
    assert "--resolution" in capsys.readouterr().err


def test_benchmark_smallest_supported_resolution_runs(benchmark, capsys):
    benchmark.main(["--resolution", str(benchmark.MIN_RESOLUTION), "--max-units", "1"])
    output = capsys.readouterr().out
    assert f"grid {benchmark.MIN_RESOLUTION}px" in output
    assert "per unit (ms)" in output
