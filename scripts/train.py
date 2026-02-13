"""Скрипт запуска обучения LLM."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.training.trainer import train


def main():
    parser = argparse.ArgumentParser(description="Обучение LLM модели")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Путь к конфигурационному файлу",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
