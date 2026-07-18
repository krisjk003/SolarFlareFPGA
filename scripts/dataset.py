"""
scripts/dataset.py

Dataset integration and dataloader construction for the SDO Benchmark project.
Edited for metadata-driven architecture.
"""

import csv
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
from torch.utils.data import DataLoader

from datasets.sdo_dataset import SDOBenchmarkDataset
from datasets.scanner import DatasetScanner, SequenceRecord
from utils.config import load_config
from utils.logger import setup_logger


EXPECTED_FRAMES = 40


def _get_config_val(config: Dict[str, Any], key: str, default=None):
    keys = key.split(".")
    val = config
    for k in keys:
        if isinstance(val, dict) and k in val:
            val = val[k]
        else:
            return default
    return val


def _get_or_setup_logger(config):
    log_dir = Path(_get_config_val(config, "paths.log_dir", "logs"))
    return setup_logger(log_dir, "dataset")


def _binary_label(peak_flux: float) -> int:
    return 1 if peak_flux >= 1e-5 else 0


def load_metadata(dataset_root: Path):
    metadata_file = dataset_root / "meta_data.csv"

    if not metadata_file.exists():
        raise FileNotFoundError(f"Missing {metadata_file}")

    metadata = {}

    with open(metadata_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required = {"id", "peak_flux"}

        if not required.issubset(reader.fieldnames):
            raise ValueError(
                f"meta_data.csv must contain columns {required}"
            )

        for row in reader:
            seq_id = row["id"].strip()

            peak_flux = float(row["peak_flux"])

            metadata[seq_id] = {
                "label": _binary_label(peak_flux),
                "peak_flux": peak_flux,
                "start": row.get("start"),
                "end": row.get("end"),
            }

    return metadata


def load_split_dataset(config, split):

    logger = _get_or_setup_logger(config)

    dataset_root = Path(config["dataset"]["root"])
    splits_dir = Path(config["paths"]["splits_dir"])

    metadata = load_metadata(dataset_root)

    csv_file = splits_dir / f"{split}.csv"

    if not csv_file.exists():
        raise FileNotFoundError(csv_file)

    image_extensions = config["dataset"]["image_extensions"]
    image_size = tuple(config["dataset"]["image_size"])
    mean = tuple(config["dataset"]["mean"])
    std = tuple(config["dataset"]["std"])
    frame_mode = config["dataset"]["frame_mode"]

    split_map = {
        "train": config["dataset"].get("train_split_name", "training"),
        "val": "validation",
        "test": "test",
    }

    split_root = dataset_root / split_map[split]

    scanner = DatasetScanner(
        dataset_root=dataset_root,
        image_extensions=image_extensions,
    )

    records: List[SequenceRecord] = []
    label_map = {}

    with open(csv_file, encoding="utf-8") as f:

        reader = csv.DictReader(f)

        id_column = None

        for candidate in ["id", "sequence_id"]:
            if candidate in reader.fieldnames:
                id_column = candidate
                break

        if id_column is None:
            raise ValueError(
                f"{csv_file} must contain 'id' or 'sequence_id'"
            )

        for row in reader:

            seq_id = row[id_column].strip()

            if seq_id not in metadata:
                raise KeyError(
                    f"{seq_id} missing from meta_data.csv"
                )

            label_map[seq_id] = metadata[seq_id]["label"]

    for seq_id in label_map:

        active_region, sequence_name = seq_id.split("/")

        seq_dir = split_root / active_region / sequence_name

        record, skipped = scanner.scan_sequence(
            active_region,
            seq_dir,
        )

        if not record.image_paths:
            raise FileNotFoundError(seq_dir)

        if len(record.image_paths) != EXPECTED_FRAMES:
            raise ValueError(
                f"{seq_id}: expected {EXPECTED_FRAMES} frames "
                f"but found {len(record.image_paths)}"
            )

        records.append(record)

    logger.info(
        f"Loaded {len(records)} sequences for {split}"
    )

    return SDOBenchmarkDataset(
        records=records,
        label_map=label_map,
        image_size=image_size,
        mean=mean,
        std=std,
        frame_mode=frame_mode,
    )


def load_train_dataset(config):
    return load_split_dataset(config, "train")


def load_validation_dataset(config):
    return load_split_dataset(config, "val")


def load_test_dataset(config):
    return load_split_dataset(config, "test")


def build_dataloader(dataset, config, split):

    batch_size = config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]

    shuffle = split == "train"

    generator = torch.Generator().manual_seed(
        config["dataset"].get("split_seed", 42)
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        generator=generator,
    )


def load_train_val_loaders(config_path):

    config = load_config(config_path)

    train_dataset = load_train_dataset(config)
    val_dataset = load_validation_dataset(config)

    return (
        build_dataloader(train_dataset, config, "train"),
        build_dataloader(val_dataset, config, "val"),
    )


def load_test_loader(config_path):

    config = load_config(config_path)

    dataset = load_test_dataset(config)

    return build_dataloader(dataset, config, "test")
