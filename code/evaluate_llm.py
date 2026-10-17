import argparse
import json
from pathlib import Path

import pandas as pd

from triage.llm import summarize
from triage.preprocessing import clean_text


def main():
    parser = argparse.ArgumentParser(description="Проверка резюме на отдельной выборке")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    categories = json.loads((root / "categories.json").read_text(encoding="utf-8"))
    data = pd.read_excel(args.input)
    if "text" not in data or data.empty or data.text.isna().any():
        raise ValueError("Нужен непустой столбец text без пропусков")
    results = [summarize(clean_text(str(value)), categories) for value in data.text]
    valid = sum(result["status"] == "draft" for result in results)
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"row": index, "status": result["status"],
                   "draft": json.dumps(result["result"], ensure_ascii=False),
                   "faithfulness_manual": None, "completeness_manual": None}
                  for index, result in enumerate(results)]).to_excel(target / "manual_review.xlsx", index=False)
    (target / "llm_metrics.json").write_text(json.dumps({
        "rows": len(results), "validated_draft_fraction": valid / len(results),
        "limitation": "Доля учитывает получение локального резюме; полноту оценивает оператор. Ошибки загрузки весов входят в отказы."
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Созданы файлы: {target / 'manual_review.xlsx'}, {target / 'llm_metrics.json'}")


if __name__ == "__main__":
    main()
