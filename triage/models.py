import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer


class TicketModel:
    def __init__(self, categories, backend="tfidf", encoder_name="intfloat/multilingual-e5-base"):
        self.categories = list(categories)
        if len(self.categories) < 2 or len(set(self.categories)) != len(self.categories):
            raise ValueError("Нужны минимум две уникальные категории")
        if any(not isinstance(category, str) or not category.strip() for category in self.categories):
            raise ValueError("Категории должны быть непустыми строками")
        if backend not in {"tfidf", "e5"}:
            raise ValueError("Неизвестный способ получения признаков")
        self.backend = backend
        self.encoder_name = encoder_name
        self.encoder = None

    def features(self, texts, fit=False):
        if self.backend == "tfidf":
            if fit:
                self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, sublinear_tf=True)
                return self.vectorizer.fit_transform(texts)
            return self.vectorizer.transform(texts)
        if self.encoder is None:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(self.encoder_name)
        return self.encoder.encode(["query: " + text for text in texts], normalize_embeddings=True)

    def fit(self, data):
        features = self.features(data.text.tolist(), fit=True)
        self.primary = LogisticRegression(max_iter=2000, random_state=42).fit(features, data.primary)
        self.urgency = LogisticRegression(max_iter=2000, random_state=42).fit(features, data.urgency)
        self.labels = MultiLabelBinarizer(classes=self.categories)
        targets = self.labels.fit_transform([
            [row.primary] + ([row.secondary] if isinstance(row.secondary, str) and row.secondary else [])
            for row in data.itertuples()
        ])
        self.topics = OneVsRestClassifier(LogisticRegression(max_iter=2000, random_state=42)).fit(features, targets)
        return self

    def predict(self, texts, threshold=0.7, secondary_threshold=0.5):
        texts = list(texts)
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Для прогноза нужны непустые строки")
        if not np.isfinite(threshold) or not 0 <= threshold <= 1.01:
            raise ValueError("Порог уверенности должен находиться между 0 и 1.01")
        if not np.isfinite(secondary_threshold) or not 0 <= secondary_threshold <= 1:
            raise ValueError("Порог дополнительной темы должен находиться между 0 и 1")
        if not texts:
            return []
        features = self.features(texts)
        probabilities = self.primary.predict_proba(features)
        topics = self.topics.predict_proba(features)
        urgency = self.urgency.predict_proba(features)
        results = []
        for index, probabilities_row in enumerate(probabilities):
            winner = int(np.argmax(probabilities_row))
            primary = str(self.primary.classes_[winner])
            alternatives = [position for position in np.argsort(topics[index])[::-1]
                            if self.categories[position] != primary and topics[index, position] >= secondary_threshold]
            confidence = float(probabilities_row[winner])
            urgency_position = int(np.argmax(urgency[index]))
            urgency_confidence = float(urgency[index, urgency_position])
            results.append({"primary": primary, "secondary": self.categories[alternatives[0]] if alternatives else None,
                            "confidence": confidence, "urgency": str(self.urgency.classes_[urgency_position]),
                            "urgency_confidence": urgency_confidence,
                            "operator_review": confidence < threshold or urgency_confidence < threshold,
                            "operator_confirmation_required": True})
        return results
