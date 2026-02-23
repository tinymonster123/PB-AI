"""生成 manifest.json，v0.2 schema。"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


ShardKind = Literal["embed", "layer", "lm_head"]


@dataclass
class Shard:
    id: str
    kind: ShardKind
    filename: str
    bytes: int
    hash: str
    layer_range: tuple[int, int] | None = None

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "kind": self.kind,
            "filename": self.filename,
            "bytes": self.bytes,
            "hash": f"blake3:{self.hash}",
        }
        if self.layer_range is not None:
            d["layer_range"] = list(self.layer_range)
        return d


@dataclass
class ModelManifest:
    model_id: str
    variant: str
    framework: str
    dtype: str
    total_layers: int
    shards: list[Shard]
    version: str = "0.2"

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "model_id": self.model_id,
            "variant": self.variant,
            "framework": self.framework,
            "dtype": self.dtype,
            "total_layers": self.total_layers,
            "shards": [s.to_dict() for s in self.shards],
        }

    def write(self, output_dir: Path) -> Path:
        path = output_dir / "manifest.json"
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"已写入 {path}")
        return path
