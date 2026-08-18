"""Same-matrix comparison of our Dawid-Skene against crowd-kit's.

crowd-kit is not a project dependency (see the script's docstring), so this skips unless it is
present. Run it with:

    uv run --with crowd-kit python -m pytest tests/protocol/test_dawid_skene_reference_validation.py

``python -m pytest`` and not bare ``pytest``: ``uv run --with`` layers an overlay environment on
top of the project one, but the ``pytest`` console script's shebang points at the base interpreter
and never sees the overlay, so the plain form silently skips.
"""

import numpy as np
import pytest

pytest.importorskip(
    "crowdkit", reason="reference implementation; run with `uv run --with crowd-kit`"
)

from scripts import validate_dawid_skene_reference as validation  # noqa: E402


def test_agrees_with_the_crowdkit_reference_within_declared_tolerances():
    report = validation.run_validation()
    assert report["failures"] == []


def test_the_comparison_catches_a_transposed_confusion_matrix():
    """A validator that cannot fail proves nothing.

    Orientation is the one difference between the two libraries that survives every accuracy
    check: transposing ``M[c, k]`` leaves the labels and the posterior untouched. Feed the
    comparison a deliberately transposed reference and it has to complain.
    """
    case = validation.build_case(np.random.default_rng(validation.SEED))
    reference = validation.run_reference(case)
    reference["confusion"] = reference["confusion"].transpose(0, 2, 1)

    row = validation.compare(case, validation.run_ours(case), reference)

    assert row["failures"]
