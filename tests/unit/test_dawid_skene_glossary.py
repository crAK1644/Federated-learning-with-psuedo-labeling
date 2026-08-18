"""Keep DAWID_SKENE_GLOSSARY.md and the emitted metrics from drifting apart.

Stage 1 of the follow-up plan is a naming agreement, and a naming agreement that lives only in a
markdown file rots the first time someone adds a metric. These tests read the glossary and the
code and compare the two sets.
"""

import re
import types
from pathlib import Path

import numpy as np

from ssfl.config import HardAggregation
from ssfl.protocols.dawid_skene import ABSTAIN, DawidSkeneFit
from ssfl.strategies import ssfl as ssfl_strategy

GLOSSARY = Path(__file__).resolve().parents[2] / "DAWID_SKENE_GLOSSARY.md"
CLASS_COUNT_KEY = "global_class_<i>_count"


def _tables(text):
    """Yield ``(headers, rows)`` for every markdown table, rows as header->cell dicts."""
    lines = [line for line in text.splitlines()]
    index = 0
    while index < len(lines) - 1:
        line, following = lines[index], lines[index + 1]
        if line.startswith("|") and set(following.replace("|", "").strip()) <= set("- :"):
            headers = [cell.strip() for cell in line.strip("|").split("|")]
            rows = []
            index += 2
            while index < len(lines) and lines[index].startswith("|"):
                cells = [cell.strip() for cell in lines[index].strip("|").split("|")]
                rows.append(dict(zip(headers, cells)))
                index += 1
            yield headers, rows
        else:
            index += 1


def _documented(column):
    keys = set()
    for headers, rows in _tables(GLOSSARY.read_text()):
        if column not in headers:
            continue
        for row in rows:
            keys.update(re.findall(r"`([^`]+)`", row[column]))
    return keys


def _emitted_dawid_skene_keys():
    """Both branches of the metrics helper: no fit, and a complete fit."""
    stub = types.SimpleNamespace(hard_aggregation=HardAggregation.dawid_skene)
    majority = types.SimpleNamespace(
        global_labels=np.array([0, 1, ABSTAIN]), valid_mask=np.array([True, True, False])
    )
    fit = DawidSkeneFit(
        status="ok",
        labels=np.array([0, 1, ABSTAIN]),
        valid_mask=np.array([True, True, False]),
        iterations=3,
        converged=True,
        stop_reason="tolerance",
        log_likelihood=-1.0,
        objective=-1.0,
        eligible_clients=4,
        excluded_clients=("client-9",),
        diagonal_fraction=0.4,
        reference_diagonal_fraction=0.5,
        majority_agreement=0.9,
        max_posterior_mean=0.8,
    )
    annotations = np.array([[0, 1, ABSTAIN], [0, ABSTAIN, ABSTAIN]], dtype=np.int8)
    without = ssfl_strategy.SSFLStrategy._dawid_skene_metrics(
        stub, None, 0.0, "warmup", majority, annotations
    )
    with_fit = ssfl_strategy.SSFLStrategy._dawid_skene_metrics(
        stub, fit, 0.1, "ok", majority, annotations
    )
    return set(without) | set(with_fit)


def _emitted_round_keys():
    """The non-``ds_`` round metrics, read out of the source so a rename cannot slip past."""
    source = Path(ssfl_strategy.__file__).read_text()
    block = source.split("metrics_out = MetricRecord(", 1)[1].split("return arrays_out", 1)[0]
    keys = set(re.findall(r'^\s+"([a-z_0-9]+)":', block, flags=re.MULTILINE))
    if 'f"global_class_{index}_count"' in block:
        keys.add(CLASS_COUNT_KEY)
    return keys


def test_every_emitted_metric_is_defined_in_the_glossary():
    documented = _documented("Metric key")
    emitted = _emitted_dawid_skene_keys() | _emitted_round_keys()

    assert emitted - documented == set()


def test_the_glossary_defines_no_metric_the_code_stopped_emitting():
    documented = _documented("Metric key")
    emitted = _emitted_dawid_skene_keys() | _emitted_round_keys()

    assert documented - emitted == set()


def test_the_status_code_table_matches_the_code():
    table = {}
    for headers, rows in _tables(GLOSSARY.read_text()):
        if headers[:2] != ["Code", "`ds_status`"]:
            continue
        for row in rows:
            (name,) = re.findall(r"`([^`]+)`", row["`ds_status`"])
            table[name] = int(row["Code"])

    assert table == ssfl_strategy.DS_STATUS_CODES
