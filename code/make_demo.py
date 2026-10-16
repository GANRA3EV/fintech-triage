import json
from pathlib import Path

import pandas as pd


def main():
    root = Path(__file__).resolve().parents[1]
    categories = json.loads((root / "categories.json").read_text(encoding="utf-8"))
    rows = []
    for month in (1, 2, 3):
        for category, description in categories.items():
            for sample in range(6):
                identifier = f"{month}-{category}-{sample}"
                rows.append({"client_id": identifier, "conversation_id": identifier,
                             "timestamp": f"2025-{month:02d}-{sample + 1:02d}",
                             "text": f"{description}. Обращение {identifier}. Прошу проверить статус.",
                             "primary": category, "secondary": None,
                             "urgency": "high" if sample % 2 else "normal"})
    target = root / "data" / "synthetic.xlsx"
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_excel(target, index=False)
    print(f"Создан синтетический набор: {target}")


if __name__ == "__main__":
    main()
