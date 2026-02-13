"""Основной модуль обучения LLM."""

from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling, AutoTokenizer

from src.models.model_loader import load_model, load_tokenizer, apply_lora
from src.training.data_processor import load_and_prepare_dataset


def create_training_args(config: dict) -> TrainingArguments:
    """Создаёт аргументы обучения из конфигурации."""
    t = config["training"]
    return TrainingArguments(
        output_dir=t["output_dir"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        weight_decay=t["weight_decay"],
        warmup_ratio=t["warmup_ratio"],
        lr_scheduler_type=t["lr_scheduler_type"],
        logging_steps=t["logging_steps"],
        save_steps=t["save_steps"],
        eval_steps=t["eval_steps"],
        eval_strategy="steps",
        save_total_limit=t["save_total_limit"],
        fp16=t["fp16"],
        bf16=t["bf16"],
        gradient_checkpointing=t["gradient_checkpointing"],
        report_to=t["report_to"],
        seed=t["seed"],
        remove_unused_columns=False,
    )


def train(config: dict):
    """Запускает полный цикл обучения."""
    print("=== Загрузка токенизатора ===")
    tokenizer = load_tokenizer(config["model"]["name"])

    print("=== Загрузка модели ===")
    model = load_model(config)

    print("=== Применение LoRA ===")
    model = apply_lora(model, config)

    print("=== Подготовка данных ===")
    datasets = load_and_prepare_dataset(config, tokenizer)

    print("=== Настройка обучения ===")
    training_args = create_training_args(config)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["eval"],
        data_collator=data_collator,
    )

    print("=== Начало обучения ===")
    trainer.train()

    print("=== Сохранение модели ===")
    trainer.save_model(config["training"]["output_dir"])
    tokenizer.save_pretrained(config["training"]["output_dir"])

    print(f"Модель сохранена в {config['training']['output_dir']}")
