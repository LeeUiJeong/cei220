from __future__ import annotations

import hashlib
import hmac
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _to_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    return str(value).encode("utf-8")


def hkdf_extract(salt: Any, input_key_material: Any, hash_name: str = "sha256") -> bytes:
    return hmac.new(_to_bytes(salt), _to_bytes(input_key_material), getattr(hashlib, hash_name)).digest()


def hkdf_expand(prk: bytes, info: Any, length: int = 8, hash_name: str = "sha256") -> bytes:
    hash_len = getattr(hashlib, hash_name)().digest_size
    if length <= 0 or length > 255 * hash_len:
        raise ValueError("HKDF output length is outside the RFC 5869 range.")

    info_bytes = _to_bytes(info)
    output = b""
    previous = b""
    counter = 1
    while len(output) < length:
        previous = hmac.new(
            prk,
            previous + info_bytes + bytes([counter]),
            getattr(hashlib, hash_name),
        ).digest()
        output += previous
        counter += 1
    return output[:length]


def derive_seed(
    external_seed: int,
    namespace: str,
    salt: str,
    *,
    bits: int = 32,
) -> int:
    if bits not in (32, 64):
        raise ValueError("Only 32-bit or 64-bit derived seeds are supported.")
    prk = hkdf_extract(salt=salt, input_key_material=external_seed)
    raw = hkdf_expand(prk=prk, info=namespace, length=bits // 8)
    return int.from_bytes(raw, byteorder="big", signed=False) % (2**bits)


def set_all_rngs(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state_fingerprint() -> str:
    digest = hashlib.sha256()
    digest.update(torch.get_rng_state().cpu().numpy().tobytes())
    numpy_state = np.random.get_state()
    digest.update(numpy_state[1].tobytes())
    python_state = random.getstate()
    digest.update(repr(python_state[1][:32]).encode("utf-8"))
    if torch.cuda.is_available():
        for state in torch.cuda.get_rng_state_all():
            digest.update(state.cpu().numpy().tobytes())
    return digest.hexdigest()


def namespace_prefix(dataset: str, model: str, namespace_version: str) -> str:
    return f"{namespace_version}/{dataset}/{model}"


def full_namespace(dataset: str, model: str, namespace_version: str, component: str) -> str:
    return f"{namespace_prefix(dataset, model, namespace_version)}/{component}"


def seed_worker(worker_id: int, worker_base_seed: int) -> None:
    worker_seed = (int(worker_base_seed) + int(worker_id)) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def make_worker_init_fn(worker_base_seed: int):
    def _worker_init(worker_id: int) -> None:
        seed_worker(worker_id, worker_base_seed)

    return _worker_init


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


@dataclass
class SeedManager:
    external_seed: int
    dataset: str
    model: str
    namespace_version: str
    salt: str
    log_path: Path

    def __post_init__(self) -> None:
        self._cache: dict[str, int] = {}
        self._event_index = 0

    def namespace(self, component: str) -> str:
        return full_namespace(self.dataset, self.model, self.namespace_version, component)

    def derive(self, component: str, target: str) -> int:
        namespace = self.namespace(component)
        if namespace not in self._cache:
            self._cache[namespace] = derive_seed(
                self.external_seed,
                namespace,
                self.salt,
            )
        seed = self._cache[namespace]
        self._write(
            {
                "event": "derive",
                "external_seed": int(self.external_seed),
                "salt": self.salt,
                "namespace": namespace,
                "derived_seed": int(seed),
                "target_component": target,
            }
        )
        return seed

    def inject(self, component: str, seed: int, target: str) -> None:
        self._write(
            {
                "event": "inject",
                "external_seed": int(self.external_seed),
                "namespace": self.namespace(component),
                "derived_seed": int(seed),
                "target_component": target,
                "rng_state_fingerprint_after_injection": rng_state_fingerprint(),
            }
        )

    def _write(self, record: dict[str, Any]) -> None:
        record = {"event_index": self._event_index, **record}
        self._event_index += 1
        _append_jsonl(self.log_path, record)
