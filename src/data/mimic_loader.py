"""MIMIC-IV Data Ingestion and Archive Management.

Locates, validates, and extracts MIMIC-IV archives, or initializes the official
PhysioNet MIMIC-IV Clinical Database Demo structure. Reports disk usage,
row counts, and verifies required relational tables.
"""

import os
import zipfile
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_TABLES = [
    "patients",
    "admissions",
    "d_labitems",
    "labevents",
    "prescriptions",
    "diagnoses_icd",
    "icustays",
    "chartevents",
]


class MimicDataLoader:
    """Manages MIMIC-IV raw archives, extraction, and schema discovery."""

    def __init__(self, raw_dir: str = "data/raw", extracted_dir: str = "data/extracted"):
        self.raw_dir = Path(raw_dir)
        self.extracted_dir = Path(extracted_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.extracted_dir.mkdir(parents=True, exist_ok=True)

    def find_mimic_archive(self) -> Optional[Path]:
        """Scans workspace and data directories for MIMIC-IV zip archives."""
        search_dirs = [self.raw_dir, Path("data"), Path("."), Path("/workspace"), Path("/tmp")]
        for d in search_dirs:
            if d.exists():
                for f in d.glob("*.zip"):
                    if "mimic" in f.name.lower():
                        return f
        # Any zip file in raw_dir
        raw_zips = list(self.raw_dir.glob("*.zip"))
        if raw_zips:
            return raw_zips[0]
        return None

    def extract_archive(self, zip_path: Path) -> List[str]:
        """Extracts the MIMIC-IV archive with integrity verification."""
        logger.info(f"Extracting MIMIC-IV archive from {zip_path}...")
        extracted_files = []
        with zipfile.ZipFile(zip_path, "r") as archive:
            # Check CRC32 integrity
            bad_file = archive.testzip()
            if bad_file:
                raise ValueError(f"Corrupt file in archive: {bad_file}")

            for member in archive.infolist():
                if member.is_dir():
                    continue
                filename = Path(member.filename).name
                target_path = self.extracted_dir / filename
                if not target_path.exists():
                    archive.extract(member, self.extracted_dir)
                    # If extracted in subfolder, flatten to extracted_dir
                    extracted_path = self.extracted_dir / member.filename
                    if extracted_path != target_path and extracted_path.exists():
                        extracted_path.rename(target_path)
                extracted_files.append(str(target_path))
        return extracted_files

    def inventory_tables(self) -> Dict[str, Dict]:
        """Scans extracted directory and catalogs available tables, sizes, and row counts."""
        inventory = {}
        for tbl in REQUIRED_TABLES:
            # Look for tbl.csv or tbl.parquet or tbl.csv.gz
            candidates = [
                self.extracted_dir / f"{tbl}.csv",
                self.extracted_dir / f"{tbl}.parquet",
                self.extracted_dir / f"{tbl}.csv.gz",
            ]
            found_path = None
            for c in candidates:
                if c.exists():
                    found_path = c
                    break

            if found_path:
                file_size_mb = round(found_path.stat().st_size / (1024 * 1024), 2)
                # Count sample rows or total
                try:
                    if found_path.suffix == ".parquet":
                        df = pd.read_parquet(found_path)
                        num_rows = len(df)
                        cols = list(df.columns)
                    else:
                        # Quick row count estimate
                        df_head = pd.read_csv(found_path, nrows=5)
                        cols = list(df_head.columns)
                        # Count total lines
                        with open(found_path, "rb") as f:
                            num_rows = sum(1 for _ in f) - 1
                except Exception as e:
                    num_rows = -1
                    cols = []

                inventory[tbl] = {
                    "path": str(found_path),
                    "size_mb": file_size_mb,
                    "row_count": num_rows,
                    "columns": cols,
                    "status": "AVAILABLE",
                }
            else:
                inventory[tbl] = {
                    "path": None,
                    "size_mb": 0.0,
                    "row_count": 0,
                    "columns": [],
                    "status": "MISSING",
                }
        return inventory
