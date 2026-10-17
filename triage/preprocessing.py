import hashlib
import html
import re

import pandas as pd


def clean_text(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html.unescape(value))
    text = re.split(r"(?im)^\s*(?:с уважением|отправлено с|sent from)\b", text)[0]
    patterns = [
        r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b",
        r"(?<!\w)\+?\d[\d ()-]{8,}\d(?!\w)",
    ]
    for pattern in patterns:
        text = re.sub(pattern, " [СКРЫТО] ", text)
    return re.sub(r"\s+", " ", text).strip()


def prepare(frame: pd.DataFrame, labeled: bool = True) -> pd.DataFrame:
    required = {"client_id", "conversation_id", "timestamp", "text"}
    if labeled:
        required |= {"primary", "secondary", "urgency"}
    if required - set(frame):
        raise ValueError(f"Отсутствуют столбцы: {sorted(required - set(frame))}")
    if frame[list(required - {"secondary"})].isna().any().any():
        raise ValueError("Пропуски в обязательных полях")
    data = frame.copy()
    for column in ("client_id", "conversation_id", "text"):
        if data[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Пустые значения поля {column}")
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="raise")
    data["text"] = data["text"].astype(str).map(clean_text)
    data = data.loc[data.text.str.len() > 0].sort_values("timestamp")
    nonempty_messages = len(data)
    rows = []
    for _, group in data.groupby(["client_id", "conversation_id"], sort=False):
        if labeled:
            for column in ("primary", "secondary", "urgency"):
                if group[column].fillna("").nunique() > 1:
                    raise ValueError("Разметка переписки противоречива")
        row = group.iloc[-1].to_dict()
        row["start"] = group.timestamp.min()
        row["text"] = " ".join(dict.fromkeys(group.text))
        row["digest"] = hashlib.sha256(row["text"].casefold().encode()).hexdigest()
        rows.append(row)
    if not rows:
        raise ValueError("После очистки отсутствуют обращения")
    prepared = pd.DataFrame(rows)
    if labeled:
        conflicts = prepared.fillna("").groupby("digest")[["primary", "secondary", "urgency"]].nunique()
        if conflicts.gt(1).any().any():
            raise ValueError("Дубликаты текста имеют противоречивую разметку")
    result = prepared.drop_duplicates("digest").reset_index(drop=True)
    result.attrs['preparation_counts'] = {
        'input_messages': len(frame), 'empty_after_cleaning': len(frame) - nonempty_messages,
        'messages_absorbed_in_threads': nonempty_messages - len(prepared),
        'duplicate_conversations': len(prepared) - len(result), 'prepared_conversations': len(result)}
    return result


def temporal_split(data: pd.DataFrame, train_end: str, validation_end: str):
    first, second = pd.Timestamp(train_end), pd.Timestamp(validation_end)
    if first.tzinfo is None:
        first = first.tz_localize("UTC")
    if second.tzinfo is None:
        second = second.tz_localize("UTC")
    if first >= second:
        raise ValueError("Границы времени заданы неверно")
    train = data.loc[data.timestamp < first].copy()
    validation = data.loc[(data.start >= first) & (data.timestamp < second)].copy()
    test = data.loc[data.start >= second].copy()
    validation = validation.loc[~validation.client_id.isin(train.client_id)]
    test = test.loc[~test.client_id.isin(set(train.client_id) | set(validation.client_id))]
    if any(part.empty for part in (train, validation, test)):
        raise ValueError("После исключения пересечений одна из выборок пуста")
    return train, validation, test
