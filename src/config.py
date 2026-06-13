from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
DATA_DIR = PROJECT_ROOT / "data"
MODEL_PATH = ARTIFACT_DIR / "piq_crossfire_model.pt"
SUMMARY_PATH = ARTIFACT_DIR / "training_summary.json"
DB_PATH = DATA_DIR / "piq_crossfire.db"

HASH_DIM = 768

QUALITY_LABELS: tuple[str, ...] = (
    "initiative",
    "responsibility",
    "teamwork",
    "communication",
    "leadership",
    "planning",
    "adaptability",
    "emotional_control",
    "self_awareness",
    "discipline",
    "resilience",
)

QUALITY_DISPLAY = {
    "initiative": "Initiative",
    "responsibility": "Responsibility",
    "teamwork": "Teamwork",
    "communication": "Communication",
    "leadership": "Leadership",
    "planning": "Planning",
    "adaptability": "Adaptability",
    "emotional_control": "Emotional Control",
    "self_awareness": "Self-Awareness",
    "discipline": "Discipline",
    "resilience": "Resilience",
}

STATUS_LABELS = {
    0: "Vague",
    1: "Usable",
    2: "Strong Evidence",
}

PIQ_FIELDS = (
    "hobbies",
    "strengths",
    "weaknesses",
    "responsibilities",
    "achievements",
    "failures",
    "friends_view",
    "family_view",
    "daily_routine",
)

CLAIM_KEYWORDS = {
    "initiative": ("initiative", "start", "started", "first", "volunteer", "proactive", "idea", "took step"),
    "responsibility": ("responsible", "responsibility", "owned", "handled", "managed", "accountable", "duty"),
    "teamwork": ("team", "group", "friends", "collaborate", "cooperate", "with others", "support"),
    "communication": ("communicate", "explain", "speak", "listen", "convince", "discussion", "express"),
    "leadership": ("lead", "leader", "guided", "coordinated", "captain", "organised", "organized"),
    "planning": ("plan", "schedule", "organise", "organize", "prepare", "strategy", "timeline"),
    "adaptability": ("adapt", "adjust", "flexible", "changed", "new situation", "learn quickly"),
    "emotional_control": ("calm", "pressure", "stress", "anger", "emotional", "composed", "patience"),
    "self_awareness": ("weakness", "improve", "realized", "learnt", "learned", "mistake", "feedback"),
    "discipline": ("discipline", "consistent", "routine", "daily", "regular", "practice", "habit"),
    "resilience": ("failed", "failure", "setback", "again", "recovered", "continued", "did not give up"),
}

GENERIC_PHRASES = (
    "hardworking",
    "confident",
    "good person",
    "team player",
    "leadership quality",
    "positive attitude",
    "always ready",
    "never give up",
)

