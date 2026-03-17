from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path
from typing import Final, TypedDict
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_URL: Final[str] = "https://raw.githubusercontent.com/docyx/pc-part-dataset/main/data/csv.zip"
DEFAULT_OUTPUT_DIR: Final[Path] = Path("data") / "pc_part_dataset"
DEFAULT_ARCHIVE_NAME: Final[str] = "csv.zip"
CHUNK_SIZE: Final[int] = 1024 * 1024

class DownloadResult(TypedDict):
    """Structured result returned by ``download_pc_part_csv``."""

    url: str
    output_dir: Path
    archive_sha256: str
    archive_deleted: bool
    csv_files: list[str]

def format_bytes(num_bytes: int) -> str:
    """Return a human-readable byte string."""
    units: tuple[str, ...] = ("B", "KB", "MB", "GB", "TB")
    size: float = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"

def sha256_file(path: Path) -> str:
    """Compute the SHA256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()

def download_file(url: str, destination: Path, timeout: int = 60) -> Path:
    """Download a file from ``url`` to ``destination`` with progress output."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    request = Request(
        url,
        headers={
            "User-Agent": "python3-download-script/1.0",
            "Accept": "*/*",
        },
    )

    try:
        with urlopen(request, timeout=timeout) as response, destination.open("wb") as out_file:
            total_header = response.headers.get("Content-Length")
            total_bytes: int = int(total_header) if total_header is not None else 0
            downloaded: int = 0

            print(f"Downloading: {url}")
            if total_bytes > 0:
                print(f"Expected size: {format_bytes(total_bytes)}")
            else:
                print("Expected size: unknown")

            while True:
                chunk: bytes = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)

                if total_bytes > 0:
                    pct: float = downloaded / total_bytes * 100.0
                    print(
                        f"  downloaded {format_bytes(downloaded)} / {format_bytes(total_bytes)} ({pct:5.1f}%)",
                        end="\r",
                        flush=True,
                    )
                else:
                    print(f"  downloaded {format_bytes(downloaded)}", end="\r", flush=True)

            print()
            print(f"Saved archive to: {destination.resolve()}")
            return destination
    except HTTPError as exc:
        raise RuntimeError(f"HTTP error while downloading {url}: {exc.code} {exc.reason}") from exc
    except URLError as exc:
        raise RuntimeError(f"Network error while downloading {url}: {exc.reason}") from exc

def extract_csvs_flat(archive_path: Path, output_dir: Path) -> list[Path]:
    """Extract all CSV files from ``archive_path`` directly into ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted_files: list[Path] = []

    with zipfile.ZipFile(archive_path, "r") as zip_file:
        csv_members: list[zipfile.ZipInfo] = [
            member
            for member in zip_file.infolist()
            if not member.is_dir() and member.filename.lower().endswith(".csv")
        ]

        if not csv_members:
            raise RuntimeError("No CSV files found in the downloaded archive.")

        print(f"Extracting {len(csv_members)} CSV files to: {output_dir.resolve()}")

        seen_names: set[str] = set()
        for member in csv_members:
            filename: str = Path(member.filename).name
            if filename in seen_names:
                raise RuntimeError(f"Duplicate CSV filename found in archive: {filename}")
            seen_names.add(filename)

            target_path: Path = output_dir / filename
            with zip_file.open(member, "r") as src, target_path.open("wb") as dst:
                while True:
                    chunk: bytes = src.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    dst.write(chunk)
            extracted_files.append(target_path)

    print("Extraction complete.")
    return sorted(extracted_files)

def download_pc_part_csv(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> DownloadResult:
    """Download and extract the CSV export into ``output_dir``."""
    resolved_output_dir: Path = Path(output_dir)
    archive_path: Path = resolved_output_dir / DEFAULT_ARCHIVE_NAME

    download_file(url=DEFAULT_URL, destination=archive_path)
    archive_sha256: str = sha256_file(archive_path)
    extracted_files: list[Path] = extract_csvs_flat(archive_path=archive_path, output_dir=resolved_output_dir)
    archive_path.unlink()

    return {
        "url": DEFAULT_URL,
        "output_dir": resolved_output_dir,
        "archive_sha256": archive_sha256,
        "archive_deleted": True,
        "csv_files": [str(path) for path in extracted_files],
    }

def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Download the CSV export from docyx/pc-part-dataset into a single output directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory where CSV files will be stored. Default: {DEFAULT_OUTPUT_DIR}",
    )
    return parser

def main(argv: list[str] | None = None) -> int:
    """Run the CLI entrypoint and return a process exit code."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        result: DownloadResult = download_pc_part_csv(output_dir=args.output_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("\nDownload summary")
    print("-" * 60)
    print(f"Output dir     : {result['output_dir']}")
    print(f"Archive SHA256 : {result['archive_sha256']}")
    print(f"Archive deleted: {result['archive_deleted']}")
    print(f"CSV file count : {len(result['csv_files'])}")

    sample_files: list[str] = result["csv_files"][:10]
    if sample_files:
        print("Sample CSV files:")
        for file_path in sample_files:
            print(f"  - {file_path}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
