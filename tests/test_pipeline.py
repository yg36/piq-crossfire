from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import QUALITY_LABELS
from src.inference import (
    analyze_incident,
    analyze_practice_answer,
    audit_claim,
    build_interview_brief,
    candidate_chunks_from_life_log,
    extract_life_log_candidates,
    load_bundle,
    story_blueprint,
    summarize_life_log_themes,
)
from src.store import IncidentRecord
from src.synthetic_data import generate_incident_dataset
from src.train import train_crossfire_model


def test_synthetic_data_shapes() -> None:
    dataset = generate_incident_dataset(n_samples=80, seed=3)
    assert len(dataset["texts"]) == 80
    assert dataset["labels"].shape == (80, len(QUALITY_LABELS))


def test_training_analysis_and_claim_audit(tmp_path: Path) -> None:
    train_crossfire_model(n_samples=512, epochs=4, batch_size=64, artifact_dir=tmp_path)
    bundle = load_bundle(tmp_path / "piq_crossfire_model.pt")
    text = "During dance practice, two people disagreed. I listened calmly, adjusted the plan, and helped the group finish the routine."
    analysis = analyze_incident(text, bundle)
    assert 0 <= analysis["strength"] <= 100
    assert analysis["qualities"]

    incident = IncidentRecord(
        id=1,
        text=text,
        strength=analysis["strength"],
        status=analysis["status"],
        qualities=analysis["qualities"],
        created_at="now",
    )
    audit = audit_claim("I can work with a team and stay calm under pressure.", [incident])
    assert audit["verdict"] in {"Supported", "Supported With Risk", "Needs One More Proof", "Weak", "Unsupported"}
    assert audit["crossfire_questions"]
    assert audit["repair_plan"]


def test_story_blueprint_answer_and_brief() -> None:
    text = (
        "During dance practice, two people disagreed. I listened calmly, adjusted the plan, "
        "and helped the group finish the routine. I learned to coordinate without becoming loud."
    )
    blueprint = story_blueprint(text)
    assert blueprint["score"] >= 60
    assert blueprint["checks"]["Personal Action"]

    answer_feedback = analyze_practice_answer(text)
    assert answer_feedback["score"] >= 60
    assert answer_feedback["flags"]

    incident = IncidentRecord(
        id=1,
        text=text,
        strength=78.0,
        status="Strong Evidence",
        qualities={"teamwork": 0.9, "communication": 0.7},
        created_at="now",
    )
    brief = build_interview_brief({}, [incident], [])
    assert "PIQ Crossfire Interview Brief" in brief
    assert "Best Stories" in brief


def test_life_log_extraction(tmp_path: Path) -> None:
    train_crossfire_model(n_samples=512, epochs=4, batch_size=64, artifact_dir=tmp_path)
    bundle = load_bundle(tmp_path / "piq_crossfire_model.pt")
    raw_log = """
    Date: 12-06-2026
    Tasks Completed:
    Implemented an improved neural network architecture for a regression problem, experimenting with network structure and training workflow.
    Discussed project requirements, expected outcomes, and future implementation direction with Nikhil Sir.

    Date: 08-06-2026
    Added the AI Intelligence Engine section within the Ansi sidebar navigation following existing architectural patterns.
    Verified route registration, sidebar visibility, navigation flow, and overall integration without impacting existing resources.
    """
    chunks = candidate_chunks_from_life_log(raw_log)
    assert len(chunks) >= 2
    candidates = extract_life_log_candidates(raw_log, bundle, max_candidates=5, min_score=20)
    assert candidates
    assert candidates[0]["import_score"] >= candidates[-1]["import_score"]
    themes = summarize_life_log_themes(candidates)
    assert themes
