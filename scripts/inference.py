"""Скрипт для генерации текста с обученной моделью."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.inference import load_trained_model, generate


def main():
    parser = argparse.ArgumentParser(description="Инференс обученной LLM")
    parser.add_argument("--base-model", type=str, required=True, help="Имя базовой модели из HuggingFace")
    parser.add_argument("--adapter-path", type=str, required=True, help="Путь к обученному LoRA адаптеру")
    parser.add_argument("--prompt", type=str, help="Промпт для генерации")
    parser.add_argument("--max-tokens", type=int, default=256, help="Максимум новых токенов")
    parser.add_argument("--temperature", type=float, default=0.7, help="Температура генерации")
    parser.add_argument("--interactive", action="store_true", help="Интерактивный режим")
    args = parser.parse_args()

    print("Загрузка модели...")
    model, tokenizer = load_trained_model(args.base_model, args.adapter_path)
    print("Модель загружена!\n")

    if args.interactive:
        print("Интерактивный режим (введите 'exit' для выхода)\n")
        while True:
            prompt = input(">>> ")
            if prompt.lower() in ("exit", "quit"):
                break
            formatted = f"### Instruction:\n{prompt}\n\n### Response:\n"
            response = generate(model, tokenizer, formatted, args.max_tokens, args.temperature)
            print(f"\n{response}\n")
    elif args.prompt:
        formatted = f"### Instruction:\n{args.prompt}\n\n### Response:\n"
        response = generate(model, tokenizer, formatted, args.max_tokens, args.temperature)
        print(response)
    else:
        print("Укажите --prompt или --interactive")


if __name__ == "__main__":
    main()
