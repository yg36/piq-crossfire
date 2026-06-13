from __future__ import annotations

import random

import numpy as np

from .config import QUALITY_LABELS
from .text_features import tokenize


CONTEXTS = (
    "during my internship",
    "during dance practice",
    "while working with a college team",
    "when my friends planned a trip",
    "during a group discussion",
    "while preparing for AFSB with a full workday",
    "when a teammate was stuck",
    "during a family responsibility",
    "when a performance went wrong",
    "while learning a difficult technical topic",
)

CHALLENGES = (
    "there was very little time",
    "two people disagreed with each other",
    "I was nervous and unsure at first",
    "the plan changed suddenly",
    "the task was unclear",
    "I made a mistake in the beginning",
    "everyone was tired",
    "we had to coordinate without much guidance",
)

RESULTS = (
    "the work was completed on time",
    "the group became clearer about the next step",
    "we avoided confusion and finished the task",
    "my confidence improved after that",
    "the team trusted me with more responsibility",
    "I learned how to stay practical under pressure",
    "the performance became smoother",
    "I corrected my mistake and changed my approach",
)

LEARNINGS = (
    "I learned to act early instead of waiting",
    "I understood the value of clear communication",
    "I realized that responsibility means owning the result",
    "I learned to stay calm before deciding",
    "I understood that planning reduces pressure",
    "I realized that feedback is useful when taken honestly",
)

QUALITY_ACTIONS = {
    "initiative": (
        "noticed the gap and started the first draft",
        "volunteered to take the first step",
        "suggested a practical idea before being asked",
        "created a small checklist to move the work forward",
    ),
    "responsibility": (
        "took ownership of the task and tracked it till completion",
        "handled the duty even when it became inconvenient",
        "accepted my part of the mistake and corrected it",
        "made sure the pending work was not left to someone else",
    ),
    "teamwork": (
        "worked with the group and divided the work fairly",
        "supported a teammate without taking credit away",
        "listened to both sides and helped the group continue",
        "adjusted my pace so the team could move together",
    ),
    "communication": (
        "explained the issue in simple words",
        "asked clear questions and confirmed the next action",
        "spoke calmly so the group understood the plan",
        "summarized the topic and listened to feedback",
    ),
    "leadership": (
        "coordinated the group and assigned clear roles",
        "guided the team when everyone was confused",
        "kept the group focused on the immediate objective",
        "led the discussion by including quieter members",
    ),
    "planning": (
        "made a timeline and broke the work into smaller steps",
        "prepared the resources before the actual task started",
        "prioritized the most urgent work first",
        "planned the sequence so we did not waste time",
    ),
    "adaptability": (
        "changed my approach when the first plan failed",
        "adjusted quickly when the situation changed",
        "learned the new process and applied it the same day",
        "accepted the constraint and found another workable method",
    ),
    "emotional_control": (
        "stayed calm even when I was nervous",
        "controlled my irritation and focused on solving the issue",
        "paused before reacting and then responded practically",
        "handled the pressure without blaming anyone",
    ),
    "self_awareness": (
        "realized my mistake and asked for feedback",
        "accepted that I had overthought the situation",
        "noted where I hesitated and worked on it later",
        "understood which part of my behavior needed improvement",
    ),
    "discipline": (
        "followed the routine consistently for several days",
        "practiced daily even when I had limited time",
        "kept a fixed slot and did not skip the task",
        "repeated the basics until my performance improved",
    ),
    "resilience": (
        "failed once but tried again with a better method",
        "continued after criticism instead of stopping",
        "recovered from the setback and completed the work",
        "used the failure as feedback for the next attempt",
    ),
}

GENERIC_TEXTS = (
    "I am very hardworking and confident. I always give my best in every situation.",
    "I am a good team player and I have leadership qualities.",
    "I never give up and I always stay positive.",
    "My friends say I am helpful and responsible.",
    "I can handle pressure very well because I am strong.",
)


def _quality_vector(qualities: list[str]) -> np.ndarray:
    label_set = set(qualities)
    return np.array([1.0 if label in label_set else 0.0 for label in QUALITY_LABELS], dtype=np.float32)


def _story_strength(text: str, qualities: list[str], detail_level: int) -> float:
    tokens = set(tokenize(text))
    action = any(word in tokens for word in {"handled", "guided", "planned", "explained", "accepted", "changed", "created", "worked"})
    result = any(word in tokens for word in {"completed", "improved", "finished", "corrected", "learned", "trusted"})
    learning = any(word in tokens for word in {"learned", "realized", "understood", "feedback"})
    people = any(word in tokens for word in {"team", "group", "friends", "teammate", "family"})
    score = 25 + 9 * len(set(qualities)) + 13 * action + 14 * result + 13 * learning + 8 * people + 7 * detail_level
    return float(np.clip(score, 5, 98))


def _status_from_strength(strength: float) -> int:
    if strength >= 76:
        return 2
    if strength >= 48:
        return 1
    return 0


def generate_incident_dataset(n_samples: int = 7000, seed: int = 42) -> dict[str, np.ndarray | list[str]]:
    rng = random.Random(seed)
    texts: list[str] = []
    labels: list[np.ndarray] = []
    strengths: list[float] = []
    statuses: list[int] = []

    quality_names = list(QUALITY_LABELS)

    for _ in range(n_samples):
        if rng.random() < 0.18:
            text = rng.choice(GENERIC_TEXTS)
            if rng.random() < 0.35:
                text += " I need to add a real example for this."
            qualities = []
            strength = rng.uniform(12, 40)
            status = 0
        else:
            quality_count = rng.choices([1, 2, 3], weights=[0.38, 0.46, 0.16], k=1)[0]
            qualities = rng.sample(quality_names, quality_count)
            context = rng.choice(CONTEXTS)
            challenge = rng.choice(CHALLENGES)
            result = rng.choice(RESULTS)
            learning = rng.choice(LEARNINGS)
            actions = [rng.choice(QUALITY_ACTIONS[quality]) for quality in qualities]
            detail_level = rng.choices([0, 1, 2], weights=[0.18, 0.45, 0.37], k=1)[0]

            if detail_level == 0:
                text = f"{context}, I {actions[0]}."
            elif detail_level == 1:
                text = f"{context}, {challenge}. I {actions[0]} and {result}."
            else:
                joined_actions = "; I ".join(actions)
                text = f"{context}, {challenge}. I {joined_actions}. As a result, {result}. {learning}."

            strength = _story_strength(text, qualities, detail_level)
            strength += rng.uniform(-7, 7)
            strength = float(np.clip(strength, 5, 98))
            status = _status_from_strength(strength)

        texts.append(text)
        labels.append(_quality_vector(qualities))
        strengths.append(strength / 100.0)
        statuses.append(status)

    return {
        "texts": texts,
        "labels": np.vstack(labels).astype(np.float32),
        "strengths": np.array(strengths, dtype=np.float32),
        "statuses": np.array(statuses, dtype=np.int64),
    }

