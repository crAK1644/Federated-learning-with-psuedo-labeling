import csv
from pathlib import Path

import numpy as np
import pytest

from ssfl.config import DataPrepConfig, NormalizationMode
from ssfl.data.archive import UnsupportedArchiveError, ensure_extracted
from ssfl.data.discovery import (
    DataDiscoveryError,
    discover_source_files,
    validate_discovery,
)
from ssfl.data.io import DataValidationError, load_source_matrix, validate_source_file
from ssfl.data.labels import LABEL_MAP, NUM_FEATURES
from ssfl.data.manifest import compute_allocation_stats, gini, js_divergence
from ssfl.data.partition import build_scenario
from ssfl.data.prepare_data import run_full, run_validate_only
from ssfl.data.sampling import sample_and_split, subset_seed
from ssfl.data.scaling import fit_scaler, reshape_eq19

FEATURE_NAMES = [f"f{i}" for i in range(NUM_FEATURES)]

SIX_CLASS_DEVICES = {3, 7}


def _write_csv(path: Path, num_rows: int, rng: np.random.Generator) -> None:
    data = rng.random((num_rows, NUM_FEATURES), dtype=np.float64) * 100
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(FEATURE_NAMES)
        writer.writerows(data.tolist())


def make_synthetic_dataset(root: Path, rows_per_file: int, seed: int = 7) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    with open(root / "device_info.csv", "w") as fh:
        fh.write("DeviceID,DeviceName\n")
        for d in range(1, 10):
            fh.write(f"{d},Device{d}\n")
    for device_id in range(1, 10):
        keys = [k for k in LABEL_MAP if k.startswith("gafgyt.") or k == "benign"]
        if device_id not in SIX_CLASS_DEVICES:
            keys += [k for k in LABEL_MAP if k.startswith("mirai.")]
        for class_key in keys:
            _write_csv(root / f"{device_id}.{class_key}.csv", rows_per_file, rng)


@pytest.fixture
def synthetic_dataset(tmp_path) -> Path:
    root = tmp_path / "raw"
    make_synthetic_dataset(root, rows_per_file=60)
    return root


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------


def test_discovery_finds_89_files(synthetic_dataset) -> None:
    files = discover_source_files(synthetic_dataset)
    assert len(files) == 89
    validate_discovery(files)  # must not raise


def test_discovery_rejects_wrong_device_count(tmp_path) -> None:
    root = tmp_path / "raw"
    make_synthetic_dataset(root, rows_per_file=10)
    (root / "10.benign.csv").write_text((root / "1.benign.csv").read_text())
    files = discover_source_files(root)
    with pytest.raises(DataDiscoveryError):
        validate_discovery(files)


def test_discovery_ignores_non_matching_files(synthetic_dataset) -> None:
    (synthetic_dataset / "features.csv").write_text("Feature Name,Feature Description\nf0,x\n")
    (synthetic_dataset / "data_summary.csv").write_text("a,b,c\n")
    files = discover_source_files(synthetic_dataset)
    assert len(files) == 89  # extra non-device CSVs are not picked up


# ---------------------------------------------------------------------------
# io validation
# ---------------------------------------------------------------------------


def test_validate_source_file_rejects_nan(tmp_path) -> None:
    root = tmp_path / "raw"
    make_synthetic_dataset(root, rows_per_file=10)
    files = discover_source_files(root)
    source = files[0]
    matrix, columns = load_source_matrix(source)
    matrix[0, 0] = np.nan
    with pytest.raises(DataValidationError):
        validate_source_file(source, matrix, columns, columns, min_rows=10)


def test_load_source_matrix_preserves_rows_that_float32_would_merge(tmp_path) -> None:
    """The real ``HH_jit_*_variance`` columns reach ~5.7e17, where a float32 step is ~3.4e10.
    Loading at float32 merged distinct rows before any scaler ran; float64 must keep them apart."""
    root = tmp_path / "raw"
    make_synthetic_dataset(root, rows_per_file=10)
    source = discover_source_files(root)[0]
    big = 5.669e17
    rows = [[big + k * 1e9] * NUM_FEATURES for k in range(10)]
    with open(source.path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(FEATURE_NAMES)
        writer.writerows(rows)

    matrix, _ = load_source_matrix(source)
    assert matrix.dtype == np.float64
    assert len(np.unique(matrix, axis=0)) == 10
    assert len(np.unique(matrix.astype(np.float32), axis=0)) < 10  # what the old loader produced


def test_validate_source_file_rejects_too_few_rows(synthetic_dataset) -> None:
    files = discover_source_files(synthetic_dataset)
    source = files[0]
    matrix, columns = load_source_matrix(source)
    with pytest.raises(DataValidationError):
        validate_source_file(source, matrix, columns, columns, min_rows=matrix.shape[0] + 1)


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------


def test_subset_seed_no_collision() -> None:
    assert subset_seed(2023, 2, 3) != subset_seed(2023, 3, 2)


def test_sample_and_split_deterministic_and_disjoint(synthetic_dataset) -> None:
    config = DataPrepConfig(input_path=synthetic_dataset, seed=2023, samples_per_subset=50)
    files = discover_source_files(synthetic_dataset)
    source = files[0]
    matrix, _ = load_source_matrix(source)

    split_a = sample_and_split(source, matrix, config)
    split_b = sample_and_split(source, matrix, config)
    assert np.array_equal(split_a.private, split_b.private)

    assert split_a.private.shape[0] == 35
    assert split_a.open.shape[0] == 5
    assert split_a.test.shape[0] == 10

    all_rows = [r.source_row for r in split_a.audit_rows]
    assert len(all_rows) == len(set(all_rows)) == 50


# ---------------------------------------------------------------------------
# scaling
# ---------------------------------------------------------------------------


def test_reshape_eq19_matches_hand_derivation() -> None:
    v = np.arange(6, dtype=np.float32).reshape(1, 6)
    out = reshape_eq19(v, rows=2, cols=3)
    expected = np.array([[[0, 2, 4], [1, 3, 5]]], dtype=np.float32)
    assert np.array_equal(out, expected)


def test_reshape_eq19_shape_115() -> None:
    v = np.arange(115, dtype=np.float32).reshape(1, 115)
    out = reshape_eq19(v)
    assert out.shape == (1, 23, 5)
    assert out[0, 0, 0] == 0
    assert out[0, 1, 0] == 1
    assert out[0, 0, 1] == 23


def test_fit_scaler_handles_constant_feature() -> None:
    x = np.zeros((10, 3), dtype=np.float32)
    x[:, 0] = np.arange(10)
    x[:, 1] = 5.0  # constant
    x[:, 2] = np.arange(10) * 2
    scaler = fit_scaler(x)
    out = scaler.transform(x)
    assert np.allclose(out[:, 1], 0.0)
    assert np.isclose(out[:, 0].min(), 0.0) and np.isclose(out[:, 0].max(), 1.0)


def test_quantile_scaler_separates_what_minmax_quantises_away() -> None:
    """The gafgyt.tcp/gafgyt.udp geometry in miniature.

    One feature spans a huge range while the two groups of interest differ by a sliver of it.
    Min-max in float32 collapses that sliver to a couple of ULPs; the rank transform turns it
    into the fraction of samples lying between the two values.
    """
    big, small = 1.5e9, 183.0
    a = np.full(500, big, dtype=np.float64)
    b = np.full(500, big + small, dtype=np.float64)
    spread = np.linspace(0.0, 3.0e9, 1000)  # gives the feature its 1.5e9-scale range
    x = np.concatenate([a, b, spread]).reshape(-1, 1)

    mm = fit_scaler(x, NormalizationMode.all_mini).transform(x)
    qt = fit_scaler(x, NormalizationMode.quantile).transform(x)

    mm_gap = abs(float(mm[0, 0]) - float(mm[500, 0]))
    qt_gap = abs(float(qt[0, 0]) - float(qt[500, 0]))
    assert mm_gap < 4 * np.spacing(np.float32(mm[0, 0]))  # a few float32 ULPs, unusable
    assert qt_gap > 0.1  # ~1000 of 2000 samples sit between the two values
    assert np.all(np.diff(qt[np.argsort(x[:, 0]), 0]) >= 0)  # transform stays monotone


def test_quantile_scaler_handles_constant_feature() -> None:
    x = np.zeros((10, 2), dtype=np.float32)
    x[:, 0] = np.arange(10)
    x[:, 1] = 5.0  # constant
    scaler = fit_scaler(x, NormalizationMode.quantile)
    out = scaler.transform(x)
    assert np.allclose(out[:, 1], 0.0)
    assert np.isclose(out[:, 0].min(), 0.0) and np.isclose(out[:, 0].max(), 1.0)


# ---------------------------------------------------------------------------
# partition
# ---------------------------------------------------------------------------


def _device_label_map() -> dict[int, list[int]]:
    devices: dict[int, list[int]] = {}
    for d in range(1, 10):
        keys = [k for k in LABEL_MAP if k.startswith("gafgyt.") or k == "benign"]
        if d not in SIX_CLASS_DEVICES:
            keys += [k for k in LABEL_MAP if k.startswith("mirai.")]
        devices[d] = [LABEL_MAP[k] for k in keys]
    return devices


@pytest.mark.parametrize("scenario,expected_clients", [(1, 27), (2, 89), (3, 89)])
def test_scenario_client_counts(scenario, expected_clients) -> None:
    devices = _device_label_map()
    assignments = build_scenario(scenario, devices, private_count=700, seed=2023, dirichlet_alpha=0.1)
    assert len(assignments) == expected_clients
    # every private example used exactly once, no example crosses devices
    seen: set[tuple[int, int, int]] = set()
    for a in assignments:
        for label, idxs in a.class_local_indices.items():
            for idx in idxs:
                key = (a.device_id, label, idx)
                assert key not in seen
                seen.add(key)
    assert len(seen) == sum(len(labels) for labels in devices.values()) * 700


def test_scenario_3_every_client_nonempty() -> None:
    devices = _device_label_map()
    assignments = build_scenario(3, devices, private_count=700, seed=2023, dirichlet_alpha=0.1)
    assert all(a.num_examples > 0 for a in assignments)


# ---------------------------------------------------------------------------
# manifest stats
# ---------------------------------------------------------------------------


def test_js_divergence_zero_for_identical() -> None:
    p = np.array([0.5, 0.5])
    assert js_divergence(p, p) == pytest.approx(0.0, abs=1e-9)


def test_gini_zero_for_equal_counts() -> None:
    assert gini(np.array([10, 10, 10, 10])) == pytest.approx(0.0, abs=1e-9)


def test_allocation_stats_shape() -> None:
    devices = _device_label_map()
    assignments = build_scenario(1, devices, private_count=700, seed=2023, dirichlet_alpha=0.1)
    stats = compute_allocation_stats(assignments, num_labels=len(LABEL_MAP))
    assert stats["num_clients"] == 27
    assert len(stats["clients"]) == 27


# ---------------------------------------------------------------------------
# end-to-end prepare_data
# ---------------------------------------------------------------------------


def test_validate_only_does_not_create_output(synthetic_dataset, tmp_path) -> None:
    output = tmp_path / "out"
    config = DataPrepConfig(
        input_path=synthetic_dataset, output_path=output, seed=2023, samples_per_subset=50, validate_only=True
    )
    report = run_validate_only(config)
    assert report["status"] == "ok"
    assert report["num_files"] == 89
    assert not output.exists()


def test_run_full_produces_expected_artifacts_and_counts(synthetic_dataset, tmp_path) -> None:
    output = tmp_path / "out"
    config = DataPrepConfig(input_path=synthetic_dataset, output_path=output, seed=2023, samples_per_subset=50)
    manifest = run_full(config)

    assert manifest["total_records"] == 89 * 50
    assert manifest["private_records"] == 89 * 35
    assert manifest["open_records"] == 89 * 5
    assert manifest["test_records"] == 89 * 10
    assert manifest["scenario_client_counts"] == {"1": "27", "2": "89", "3": "89"} or manifest[
        "scenario_client_counts"
    ] == {"1": 27, "2": 89, "3": 89}

    assert (output / "dataset_manifest.json").exists()
    assert (output / "feature_schema.json").exists()
    assert (output / "label_map.json").exists()
    assert (output / "scaler.npz").exists()
    assert (output / "audit" / "source_rows.parquet").exists()
    for scenario in (1, 2, 3):
        assert (output / "scenarios" / f"{scenario}.json").exists()
        assert (output / "plots" / f"scenario_{scenario}_allocation.png").exists()

    open_features = np.load(output / "open" / "features.npy")
    assert open_features.shape == (89 * 5, 115)
    test_labels = np.load(output / "test" / "labels.npy")
    assert test_labels.shape == (89 * 10,)

    private_files = list((output / "private").glob("*.npz"))
    assert len(private_files) == 89


def test_run_full_is_reproducible(synthetic_dataset, tmp_path) -> None:
    config_a = DataPrepConfig(
        input_path=synthetic_dataset, output_path=tmp_path / "out_a", seed=2023, samples_per_subset=50
    )
    config_b = DataPrepConfig(
        input_path=synthetic_dataset, output_path=tmp_path / "out_b", seed=2023, samples_per_subset=50
    )
    manifest_a = run_full(config_a)
    manifest_b = run_full(config_b)
    assert manifest_a["manifest_hash"] == manifest_b["manifest_hash"]


def test_run_full_rerun_swaps_atomically(synthetic_dataset, tmp_path) -> None:
    output = tmp_path / "out"
    config = DataPrepConfig(input_path=synthetic_dataset, output_path=output, seed=2023, samples_per_subset=50)
    run_full(config)
    assert (output / "dataset_manifest.json").exists()
    run_full(config)  # rerun must not fail or leave partial state
    assert (output / "dataset_manifest.json").exists()
    assert not (output.parent / f"{output.name}.previous").exists()
    assert not (output.parent / f"{output.name}.building").exists()


def test_ensure_extracted_reports_missing_path_not_archive_type(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        ensure_extracted(tmp_path / "absent", tmp_path / "extract")


def test_ensure_extracted_still_rejects_unknown_suffix(tmp_path) -> None:
    bogus = tmp_path / "data.tar.gz"
    bogus.write_bytes(b"not a zip")
    with pytest.raises(UnsupportedArchiveError):
        ensure_extracted(bogus, tmp_path / "extract")
