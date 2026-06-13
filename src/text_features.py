from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
import math
import re

import numpy as np

from .config import HASH_DIM


TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z']+")

ACTION_WORDS = {
    "helped",
    "handled",
    "led",
    "guided",
    "planned",
    "organized",
    "organised",
    "explained",
    "resolved",
    "improved",
    "started",
    "created",
    "managed",
    "coordinated",
    "practiced",
    "adapted",
    "apologized",
    "listened",
}

RESULT_WORDS = {
    "result",
    "outcome",
    "completed",
    "finished",
    "reduced",
    "improved",
    "solved",
    "fix",
    "fixed",
    "success",
    "selected",
    "performed",
    "delivered",
    "changed",
}

LEARNING_WORDS = {
    "learned",
    "learnt",
    "realized",
    "understood",
    "feedback",
    "mistake",
    "improve",
    "reflection",
}

PEOPLE_WORDS = {
    "team",
    "friend",
    "friends",
    "mentor",
    "teacher",
    "sir",
    "ma'am",
    "family",
    "group",
    "classmate",
    "colleague",
}

VAGUE_WORDS = {
    "very",
    "always",
    "never",
    "good",
    "best",
    "hardworking",
    "confident",
    "positive",
    "nice",
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(normalize_text(text))


def _hash_token(token: str, dim: int) -> tuple[int, float]:
    digest = blake2b(token.encode("utf-8"), digest_size=8).digest()
    number = int.from_bytes(digest, "little", signed=False)
    return number % dim, 1.0 if (number >> 63) == 0 else -1.0


def _ngrams(tokens: list[str]) -> list[str]:
    features: list[str] = []
    for token in tokens:
        features.append(token)
    for index in range(len(tokens) - 1):
        features.append(f"{tokens[index]}_{tokens[index + 1]}")
    for index in range(len(tokens) - 2):
        features.append(f"{tokens[index]}_{tokens[index + 1]}_{tokens[index + 2]}")
    return features


def auxiliary_features(text: str) -> np.ndarray:
    tokens = tokenize(text)
    token_count = max(len(tokens), 1)
    token_set = set(tokens)
    sentences = max(len(re.findall(r"[.!?]", text)), 1)
    numbers = len(re.findall(r"\d+", text))
    first_person = sum(token in {"i", "me", "my", "mine"} for token in tokens)

    values = np.array(
        [
            min(token_count / 80.0, 1.5),
            min(sentences / 5.0, 1.5),
            min(numbers / 3.0, 1.0),
            first_person / token_count,
            len(token_set & ACTION_WORDS) / 8.0,
            len(token_set & RESULT_WORDS) / 8.0,
            len(token_set & LEARNING_WORDS) / 8.0,
            len(token_set & PEOPLE_WORDS) / 8.0,
            len(token_set & VAGUE_WORDS) / 8.0,
            1.0 if any(word in normalize_text(text) for word in ("because", "therefore", "so that", "after that")) else 0.0,
        ],
        dtype=np.float32,
    )
    return values


@dataclass(frozen=True)
class TextFeatureExtractor:
    hash_dim: int = HASH_DIM

    @property
    def output_dim(self) -> int:
        return self.hash_dim + 10

    def transform_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.hash_dim, dtype=np.float32)
        features = _ngrams(tokenize(text))
        for feature in features:
            index, sign = _hash_token(feature, self.hash_dim)
            vector[index] += sign

        norm = math.sqrt(float(np.dot(vector, vector)))
        if norm > 0:
            vector /= norm
        return np.concatenate([vector, auxiliary_features(text)]).astype(np.float32)

    def transform(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self.transform_one(text) for text in texts]).astype(np.float32)
