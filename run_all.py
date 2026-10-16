import argparse
import json
import logging
from pathlib import Path

import joblib
import pandas as pd

from triage.experiment import train
from triage.llm import summarize
from triage.preprocessing import prepare


ROOT = Path(__file__).resolve().parent


def main():
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(filename=ROOT / "logs" / "execution.log", encoding="utf-8", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Обработка обращений для оператора")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--input", required=True)
    training.add_argument("--output", default=str(ROOT / "artifacts"))
    training.add_argument("--categories", default=str(ROOT / "categories.json"))
    training.add_argument("--train-end", required=True)
    training.add_argument("--validation-end", required=True)
    training.add_argument("--backend", choices=["tfidf", "e5"], default="tfidf")
    prediction = commands.add_parser("predict")
    prediction.add_argument("--model", required=True)
    prediction.add_argument("--input", required=True)
    prediction.add_argument("--output", required=True)
    prediction.add_argument("--with-llm", action="store_true")
    args = parser.parse_args()
    logging.info("Запуск команды %s", args.command)
    if args.command == "train":
        metrics = train(args.input, args.output, args.categories, args.train_end, args.validation_end, args.backend)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        print("Созданы файлы:")
        for path in sorted(Path(args.output).iterdir()):
            print(path)
    else:
        bundle = joblib.load(args.model)
        data = prepare(pd.read_excel(args.input, dtype={"client_id": str, "conversation_id": str}), labeled=False)
        results = bundle["model"].predict(data.text.tolist(), threshold=bundle["threshold"])
        for result, row in zip(results, data.itertuples()):
            result["digest"] = row.digest
            if args.with_llm:
                result["draft"] = summarize(row.text, bundle["categories"])
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_excel(target, index=False)
        print(f"Создан файл: {target}")
    logging.info("Команда завершена успешно")


if __name__ == "__main__":
    main()
