from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SimpleVLAConfig:
    hidden_dim: int = 256
    max_action_len: int = 16


class SimpleVLAModel:
    """Minimal placeholder model class."""

    def __init__(self, config: SimpleVLAConfig) -> None:
        self.config = config

    def train_step(self) -> float:
        return 0.01

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"hidden_dim={self.config.hidden_dim},max_action_len={self.config.max_action_len}\\n")
