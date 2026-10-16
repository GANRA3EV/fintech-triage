import json
import re
from collections import Counter
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    text = (root / "README.md").read_text(encoding="utf-8")
    sections = re.split(r"(?m)^## ", text)
    counts = {}
    for section in sections:
        lines = section.splitlines()
        counts[lines[0].strip("# ")] = len(re.findall(r"\b[\w-]+\b", "\n".join(lines[1:])))
    words = re.findall(r"\b[а-яё]+\b", text.lower())
    frequency = Counter(words)
    report = {"document": "README.md", "total_words": len(re.findall(r"\b[\w-]+\b", text)),
              "sections": counts, "frequent_words": frequency.most_common(20),
              "limitations": "Технические команды входят в подсчёт; лемматизация не применяется"}
    target = root / "logs" / "word_count_report.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Создан файл: {target}")


if __name__ == "__main__":
    main()
