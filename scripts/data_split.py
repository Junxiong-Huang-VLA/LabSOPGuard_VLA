from __future__ import annotations

import argparse
import random

from project_name.common.config import load_yaml
from project_name.common.io_utils import read_jsonl, write_jsonl
from project_name.common.logging_utils import setup_logger


def main() -> int:
    parser = argparse.ArgumentParser(description="Split SOP dataset")
    parser.add_argument("--config", default="configs/data/dataset.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    ratio = cfg["dataset"]["split_ratio"]
    rows = read_jsonl(cfg["dataset"]["annotation_file"])
    random.Random(args.seed).shuffle(rows)

    n = len(rows)
    n_train = int(n * ratio["train"])
    n_val = int(n * ratio["val"])

    train = rows[:n_train]
    val = rows[n_train : n_train + n_val]
    test = rows[n_train + n_val :]

    write_jsonl("data/splits/train.jsonl", train)
    write_jsonl("data/splits/val.jsonl", val)
    write_jsonl("data/splits/test.jsonl", test)

    logger = setup_logger("data_split")
    logger.info("split: train=%d val=%d test=%d", len(train), len(val), len(test))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
