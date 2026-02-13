"""Подготовка и обработка данных для обучения LLM."""

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer


PROMPT_TEMPLATE = """### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

PROMPT_TEMPLATE_NO_INPUT = """### Instruction:
{instruction}

### Response:
{output}"""


def format_example(example: dict) -> str:
    """Форматирует один пример в текстовый промпт."""
    if example.get("input", "").strip():
        return PROMPT_TEMPLATE.format(
            instruction=example["instruction"],
            input=example["input"],
            output=example["output"],
        )
    return PROMPT_TEMPLATE_NO_INPUT.format(
        instruction=example["instruction"],
        output=example["output"],
    )


def load_and_prepare_dataset(config: dict, tokenizer: AutoTokenizer) -> dict:
    """Загружает датасет и подготавливает его для обучения."""
    data_config = config["data"]
    max_length = data_config["max_seq_length"]

    dataset = load_dataset(data_config["dataset_name"], split=data_config["train_split"])

    def tokenize(example):
        text = format_example(example)
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

    split = tokenized_dataset.train_test_split(test_size=data_config["test_size"], seed=42)

    return {"train": split["train"], "eval": split["test"]}


def load_custom_dataset(file_path: str, tokenizer: AutoTokenizer, max_length: int = 512) -> Dataset:
    """Загружает пользовательский датасет из JSONL файла.

    Ожидаемый формат каждой строки:
    {"instruction": "...", "input": "...", "output": "..."}
    """
    dataset = load_dataset("json", data_files=file_path, split="train")

    def tokenize(example):
        text = format_example(example)
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    return dataset.map(tokenize, remove_columns=dataset.column_names)
