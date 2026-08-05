"""Optional archive extraction: lets ``--input`` point at a ZIP/RAR bundle instead of an
already-extracted directory. The real dataset copy used for this build is already a flat
directory of CSVs, so this path is exercised only defensively / by tests with synthetic archives.
"""

from __future__ import annotations

import zipfile
from pathlib import Path


class UnsupportedArchiveError(RuntimeError):
    pass


def ensure_extracted(input_path: Path, extract_dir: Path) -> Path:
    """Return a directory of files for ``input_path``: unchanged if it's already a directory,
    otherwise extracted (ZIP via stdlib) into ``extract_dir``."""
    if input_path.is_dir():
        return input_path

    if not input_path.exists():
        raise FileNotFoundError(
            f"input path {input_path} does not exist; pass a directory of N-BaIoT CSVs "
            "or a .zip archive of them"
        )

    suffix = input_path.suffix.lower()
    if suffix == ".zip":
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(input_path) as zf:
            zf.extractall(extract_dir)
        return extract_dir

    if suffix == ".rar":
        raise UnsupportedArchiveError(
            f"RAR extraction is not bundled with this project (no stdlib RAR support): "
            f"install `unrar`/`unar` plus the `rarfile` package, or extract {input_path} "
            "manually and pass the resulting directory as --input."
        )

    raise UnsupportedArchiveError(f"unsupported archive type for {input_path}")
