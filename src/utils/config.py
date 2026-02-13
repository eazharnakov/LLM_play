"""Утилиты для работы с конфигурацией."""

import yaml
from pathlib import Path


def load_config(config_path: str) -> dict:
    """Загружает YAML конфигурацию из файла."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Конфиг не найден: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def save_config(config: dict, save_path: str):
    """Сохраняет конфигурацию в YAML файл."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
