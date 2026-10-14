from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .seeds import make_worker_init_fn


def dataset_spec(name: str) -> dict[str, Any]:
    if name == "cifar10":
        return {"in_channels": 3, "num_classes": 10, "image_size": 32}
    if name == "fashion_mnist":
        return {"in_channels": 1, "num_classes": 10, "image_size": 28}
    raise ValueError(f"Unsupported dataset: {name}")


def build_transforms(name: str):
    if name == "cifar10":
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.ToTensor()
        return train_transform, test_transform
    if name == "fashion_mnist":
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(28, padding=2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.ToTensor()
        return train_transform, test_transform
    raise ValueError(f"Unsupported dataset: {name}")


def load_datasets(name: str, root: str | Path, download: bool = True):
    train_transform, test_transform = build_transforms(name)
    root = str(root)
    if name == "cifar10":
        train = datasets.CIFAR10(root=root, train=True, download=download, transform=train_transform)
        test = datasets.CIFAR10(root=root, train=False, download=download, transform=test_transform)
        return train, test
    if name == "fashion_mnist":
        train = datasets.FashionMNIST(root=root, train=True, download=download, transform=train_transform)
        test = datasets.FashionMNIST(root=root, train=False, download=download, transform=test_transform)
        return train, test
    raise ValueError(f"Unsupported dataset: {name}")


def build_loaders(
    dataset_name: str,
    data_root: str | Path,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    train_seed: int,
    worker_seed: int,
    eval_seed: int,
):
    train_dataset, test_dataset = load_datasets(dataset_name, data_root, download=True)

    train_generator = torch.Generator()
    train_generator.manual_seed(int(train_seed))
    eval_generator = torch.Generator()
    eval_generator.manual_seed(int(eval_seed))

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=False,
        generator=train_generator,
        worker_init_fn=make_worker_init_fn(worker_seed) if int(num_workers) > 0 else None,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=False,
        generator=eval_generator,
        worker_init_fn=make_worker_init_fn(worker_seed + 1_000_000) if int(num_workers) > 0 else None,
    )
    return train_loader, test_loader


def _dataset_paths(name: str, root: Path) -> list[Path]:
    if name == "cifar10":
        candidates = [root / "cifar-10-batches-py", root / "cifar-10-python.tar.gz"]
    elif name == "fashion_mnist":
        candidates = [root / "FashionMNIST" / "raw"]
    else:
        raise ValueError(name)

    files: list[Path] = []
    for candidate in candidates:
        if candidate.is_file():
            files.append(candidate)
        elif candidate.is_dir():
            files.extend(sorted(path for path in candidate.rglob("*") if path.is_file()))
    return files


def create_dataset_manifest(name: str, root: str | Path, output_path: str | Path) -> dict[str, Any]:
    root_path = Path(root).resolve()
    train, test = load_datasets(name, root_path, download=True)
    file_records: list[dict[str, Any]] = []
    digest = hashlib.sha256()

    for path in _dataset_paths(name, root_path):
        relative = path.relative_to(root_path).as_posix()
        file_hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                file_hasher.update(chunk)
        file_digest = file_hasher.hexdigest()
        record = {"path": relative, "size": path.stat().st_size, "sha256": file_digest}
        file_records.append(record)
        digest.update(relative.encode("utf-8"))
        digest.update(file_digest.encode("utf-8"))

    payload = {
        "dataset": name,
        "train_size": len(train),
        "test_size": len(test),
        "classes": list(getattr(train, "classes", [])),
        "files": file_records,
        "dataset_fingerprint": digest.hexdigest(),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return payload
