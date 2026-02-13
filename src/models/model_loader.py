"""Загрузка и настройка LLM моделей."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def load_tokenizer(model_name: str) -> AutoTokenizer:
    """Загружает токенизатор для указанной модели."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(config: dict) -> AutoModelForCausalLM:
    """Загружает базовую модель с опциональной квантизацией."""
    model_config = config["model"]

    kwargs = {
        "pretrained_model_name_or_path": model_config["name"],
        "device_map": "auto",
        "trust_remote_code": True,
    }

    if model_config.get("torch_dtype") == "bfloat16":
        kwargs["torch_dtype"] = torch.bfloat16
    elif model_config.get("torch_dtype") == "float16":
        kwargs["torch_dtype"] = torch.float16

    if model_config.get("load_in_4bit"):
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(**kwargs)
    return model


def apply_lora(model: AutoModelForCausalLM, config: dict) -> AutoModelForCausalLM:
    """Применяет LoRA адаптер к модели для эффективного fine-tuning."""
    lora_config = config["lora"]

    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=lora_config["r"],
        lora_alpha=lora_config["lora_alpha"],
        lora_dropout=lora_config["lora_dropout"],
        target_modules=lora_config["target_modules"],
        bias=lora_config["bias"],
        task_type=lora_config["task_type"],
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model
