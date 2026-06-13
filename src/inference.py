from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import (
    CLAIM_KEYWORDS,
    GENERIC_PHRASES,
    MODEL_PATH,
    QUALITY_DISPLAY,
    QUALITY_LABELS,
    STATUS_LABELS,
)
from .model import PIQCrossfireNet
from .store import IncidentRecord
from .text_features import TextFeatureExtractor, normalize_text
from .train import train_crossfire_model


ACTION_MARKERS = (
    "i helped",
    "i handled",
    "i led",
    "i guided",
    "i planned",
    "i created",
    "i explained",
    "i listened",
    "i adjusted",
    "i changed",
    "i accepted",
    "i practiced",
    "i coordinated",
    "i managed",
    "i took",
    "i started",
    "i made",
    "i asked",
)

RESULT_MARKERS = (
    "as a result",
    "result",
    "completed",
    "finished",
    "improved",
    "solved",
    "fixed",
    "changed",
    "trusted",
    "avoided",
    "helped",
)

LEARNING_MARKERS = (
    "i learned",
    "i learnt",
    "i realized",
    "i understood",
    "feedback",
    "mistake",
    "changed my approach",
    "worked on",
)

PRESSURE_MARKERS = (
    "pressure",
    "nervous",
    "disagreed",
    "confusing",
    "little time",
    "deadline",
    "stuck",
    "failed",
    "missed",
    "difficult",
    "tired",
)

PEOPLE_MARKERS = (
    "friend",
    "friends",
    "team",
    "teammate",
    "group",
    "family",
    "mentor",
    "teacher",
    "sir",
    "ma'am",
    "people",
)

LOG_ACTION_HINTS = (
    "worked",
    "built",
    "made",
    "developed",
    "implemented",
    "fixed",
    "resolved",
    "learned",
    "studied",
    "explored",
    "guided",
    "helped",
    "discussed",
    "presented",
    "coordinated",
    "planned",
    "trained",
    "analyzed",
    "reviewed",
    "tested",
    "debugged",
    "improved",
    "integrated",
    "designed",
)

LOG_NOISE_PREFIXES = (
    "date:",
    "name:",
    "team:",
    "department:",
    "department/team:",
    "tasks completed:",
    "work updates:",
    "pending actions:",
    "plan for tomorrow:",
    "eod",
    "end of the day",
)


INCIDENT_KEYWORD_BOOSTS = {
    "initiative": ("noticed", "volunteered", "started", "suggested", "created", "first step"),
    "responsibility": ("owned", "ownership", "responsibility", "handled", "managed", "duty", "completed"),
    "teamwork": ("team", "group", "teammate", "friends", "with others", "together", "both sides", "helped them"),
    "communication": ("explained", "listened", "asked", "summarized", "discussion", "spoke", "clear words"),
    "leadership": ("led", "leadership", "guided", "coordinated", "assigned", "included", "formation"),
    "planning": ("planned", "schedule", "timeline", "budget", "route", "checklist", "sequence"),
    "adaptability": ("adjusted", "changed", "adapted", "new method", "alternate", "constraint"),
    "emotional_control": ("calm", "pressure", "nervous", "irritation", "paused", "without blaming"),
    "self_awareness": ("realized", "mistake", "feedback", "overthought", "hesitated", "learned that"),
    "discipline": ("routine", "daily", "practiced", "practice", "fixed slot", "consistently"),
    "resilience": ("failed", "tried again", "continued", "setback", "criticism", "recovered"),
}


def story_blueprint(text: str) -> dict[str, Any]:
    normalized = normalize_text(text)
    word_count = len(normalized.split())
    checks = {
        "Situation": word_count >= 12 and any(marker in normalized for marker in ("during", "when", "while", "in a", "at ")),
        "Pressure": any(marker in normalized for marker in PRESSURE_MARKERS),
        "Personal Action": any(marker in normalized for marker in ACTION_MARKERS),
        "People Context": any(marker in normalized for marker in PEOPLE_MARKERS),
        "Result": any(marker in normalized for marker in RESULT_MARKERS),
        "Learning": any(marker in normalized for marker in LEARNING_MARKERS),
    }
    weights = {
        "Situation": 15,
        "Pressure": 15,
        "Personal Action": 25,
        "People Context": 10,
        "Result": 20,
        "Learning": 15,
    }
    score = sum(weights[name] for name, present in checks.items() if present)
    missing = [name for name, present in checks.items() if not present]
    if score >= 82:
        label = "Interview Ready"
    elif score >= 62:
        label = "Usable With Repair"
    elif score >= 38:
        label = "Needs Structure"
    else:
        label = "Too Vague"
    return {
        "score": int(score),
        "label": label,
        "checks": checks,
        "missing": missing,
        "word_count": word_count,
    }


def _apply_keyword_boosts(text: str, probabilities: np.ndarray) -> np.ndarray:
    normalized = normalize_text(text)
    boosted = probabilities.copy()
    for index, quality in enumerate(QUALITY_LABELS):
        hits = sum(1 for keyword in INCIDENT_KEYWORD_BOOSTS[quality] if keyword in normalized)
        if hits:
            boosted[index] = max(boosted[index], min(0.42 + hits * 0.16, 0.92))
    return boosted


def ensure_bundle(model_path: Path = MODEL_PATH) -> dict[str, Any]:
    if not model_path.exists():
        train_crossfire_model()
    return load_bundle(model_path)


def load_bundle(model_path: Path = MODEL_PATH) -> dict[str, Any]:
    artifact = torch.load(model_path, map_location="cpu", weights_only=False)
    model = PIQCrossfireNet(**artifact["model_config"])
    model.load_state_dict(artifact["model_state"])
    model.eval()
    extractor = TextFeatureExtractor(hash_dim=int(artifact["hash_dim"]))
    return {
        "model": model,
        "extractor": extractor,
        "metrics": artifact.get("metrics", {}),
        "history": artifact.get("history", []),
    }


def split_incidents(raw_text: str) -> list[str]:
    chunks = [chunk.strip(" -\t") for chunk in re.split(r"\n+|(?:\r\n)+", raw_text) if chunk.strip()]
    if len(chunks) <= 1 and len(raw_text) > 350:
        chunks = [chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+", raw_text) if len(chunk.strip()) > 30]
    return [chunk for chunk in chunks if len(chunk.split()) >= 4]


def clean_life_log_text(raw_text: str) -> str:
    replacements = {
        "\u00e2\u20ac\u201c": "-",
        "\u00e2\u20ac\u2122": "'",
        "\u00e2\u20ac\u0153": '"',
        "\u00e2\u20ac\u009d": '"',
        "\u00e2\u20ac\u00a2": "-",
        "\u00c3\u00a2": "",
        "\ufeff": "",
    }
    cleaned = raw_text
    for source, target in replacements.items():
        cleaned = cleaned.replace(source, target)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _is_noise_line(line: str) -> bool:
    normalized = normalize_text(line).strip(" :-")
    if not normalized:
        return True
    if normalized.startswith(LOG_NOISE_PREFIXES):
        return True
    if re.fullmatch(r"(day\s*)?\d{1,2}[-/:]\d{1,2}[-/:]\d{2,4}", normalized):
        return True
    if re.fullmatch(r"day\s+\d+.*", normalized) and len(normalized.split()) <= 7:
        return True
    if normalized in {"show more", "show less", "you:"}:
        return True
    return False


def _candidate_has_signal(text: str) -> bool:
    normalized = normalize_text(text)
    has_action = any(hint in normalized for hint in LOG_ACTION_HINTS)
    has_context = any(marker in normalized for marker in ("during", "while", "when", "with ", "on ", "in "))
    has_person = bool(re.search(r"\b(i|my|me|team|friend|teammate|mentor|sir|group)\b", normalized))
    return has_action and (has_context or has_person)


def candidate_chunks_from_life_log(raw_text: str) -> list[str]:
    cleaned = clean_life_log_text(raw_text)
    lines = [line.strip(" -•\t") for line in cleaned.splitlines()]
    chunks: list[str] = []
    buffer: list[str] = []

    def flush_buffer() -> None:
        nonlocal buffer
        if buffer:
            joined = " ".join(buffer).strip()
            if len(joined.split()) >= 7 and _candidate_has_signal(joined):
                chunks.append(joined)
            buffer = []

    for line in lines:
        if _is_noise_line(line):
            flush_buffer()
            continue
        if len(line.split()) < 4:
            flush_buffer()
            continue

        if _candidate_has_signal(line):
            flush_buffer()
            chunks.append(line)
        else:
            buffer.append(line)
            if len(" ".join(buffer).split()) >= 34:
                flush_buffer()

    flush_buffer()

    paragraph_candidates = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", cleaned)
        if len(paragraph.split()) >= 18 and _candidate_has_signal(paragraph)
    ]
    chunks.extend(paragraph_candidates)

    deduped: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        compact = re.sub(r"\W+", "", normalize_text(chunk))[:160]
        if compact and compact not in seen:
            seen.add(compact)
            deduped.append(chunk)
    return deduped


def _life_log_import_score(text: str, analysis: dict[str, Any]) -> float:
    blueprint = analysis["blueprint"]
    quality_count = sum(1 for value in analysis["qualities"].values() if value >= 0.45)
    risk_penalty = min(len(analysis["risk_flags"]) * 5.0, 18.0)
    length_bonus = 6.0 if 16 <= len(text.split()) <= 80 else 0.0
    score = (
        0.46 * float(analysis["strength"])
        + 0.34 * float(blueprint["score"])
        + min(quality_count * 4.0, 14.0)
        + length_bonus
        - risk_penalty
    )
    return round(max(0.0, min(100.0, score)), 1)


def extract_life_log_candidates(
    raw_text: str,
    bundle: dict[str, Any],
    *,
    max_candidates: int = 24,
    min_score: float = 36.0,
) -> list[dict[str, Any]]:
    chunks = candidate_chunks_from_life_log(raw_text)
    candidates: list[dict[str, Any]] = []
    for chunk in chunks:
        analysis = analyze_incident(chunk, bundle)
        import_score = _life_log_import_score(chunk, analysis)
        if import_score < min_score:
            continue
        candidates.append(
            {
                "text": chunk,
                "import_score": import_score,
                "strength": analysis["strength"],
                "status": analysis["status"],
                "qualities": analysis["qualities"],
                "top_qualities": analysis["top_qualities"],
                "blueprint": analysis["blueprint"],
                "risk_flags": analysis["risk_flags"],
                "analysis": analysis,
            }
        )

    candidates.sort(key=lambda item: (item["import_score"], item["strength"]), reverse=True)
    return candidates[:max_candidates]


def summarize_life_log_themes(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for quality in QUALITY_LABELS:
        related = [
            candidate
            for candidate in candidates
            if candidate["qualities"].get(quality, 0.0) >= 0.38
        ]
        if not related:
            continue
        avg_score = sum(candidate["import_score"] for candidate in related) / len(related)
        rows.append(
            {
                "Theme": QUALITY_DISPLAY[quality],
                "Moments": len(related),
                "Average Signal": round(avg_score, 1),
                "Best Moment": round(max(candidate["import_score"] for candidate in related), 1),
                "key": quality,
            }
        )
    rows.sort(key=lambda item: (item["Moments"], item["Average Signal"]), reverse=True)
    return rows


def analyze_incident(text: str, bundle: dict[str, Any]) -> dict[str, Any]:
    extractor: TextFeatureExtractor = bundle["extractor"]
    model: PIQCrossfireNet = bundle["model"]
    x = torch.tensor(extractor.transform([text]), dtype=torch.float32)
    with torch.inference_mode():
        quality_logits, strength_unit, status_logits = model(x)
        quality_probs = torch.sigmoid(quality_logits).squeeze(0).numpy()
        status_probs = torch.softmax(status_logits, dim=1).squeeze(0).numpy()

    quality_probs = _apply_keyword_boosts(text, quality_probs)
    qualities = {
        label: round(float(probability), 4)
        for label, probability in zip(QUALITY_LABELS, quality_probs)
        if probability >= 0.28
    }
    if not qualities:
        top_indices = np.argsort(quality_probs)[-2:][::-1]
        qualities = {QUALITY_LABELS[index]: round(float(quality_probs[index]), 4) for index in top_indices}

    status_id = int(np.argmax(status_probs))
    strength = round(float(strength_unit.squeeze(0).item() * 100.0), 1)
    blueprint = story_blueprint(text)
    adjusted_strength = round(0.72 * strength + 0.28 * blueprint["score"], 1)
    return {
        "text": text,
        "qualities": qualities,
        "top_qualities": sorted(qualities.items(), key=lambda item: item[1], reverse=True)[:4],
        "strength": adjusted_strength,
        "model_strength": strength,
        "blueprint": blueprint,
        "status": STATUS_LABELS[status_id],
        "status_confidence": round(float(status_probs[status_id]), 4),
        "risk_flags": incident_risk_flags(text, adjusted_strength, qualities, blueprint),
    }


def incident_risk_flags(
    text: str,
    strength: float,
    qualities: dict[str, float],
    blueprint: dict[str, Any] | None = None,
) -> list[str]:
    normalized = normalize_text(text)
    blueprint = blueprint or story_blueprint(text)
    flags: list[str] = []
    if strength < 40:
        flags.append("This sounds like a claim. Add context, action, result, and learning.")
    if any(phrase in normalized for phrase in GENERIC_PHRASES):
        flags.append("Generic phrase detected. Replace it with a concrete incident.")
    if not re.search(r"\b(i|my|me)\b", normalized):
        flags.append("Your personal role is unclear.")
    if not any(word in normalized for word in ("result", "completed", "improved", "changed", "solved", "fix", "fixed", "learned", "realized")):
        flags.append("Outcome or learning is missing.")
    if "Personal Action" in blueprint["missing"]:
        flags.append("Your exact action is not sharp enough.")
    if "Learning" in blueprint["missing"]:
        flags.append("Add what changed in you after this incident.")
    if max(qualities.values(), default=0.0) < 0.38:
        flags.append("The evidence signal is weak. Make the action more specific.")
    return flags


def infer_claim_qualities(claim: str) -> list[str]:
    normalized = normalize_text(claim)
    matched: list[str] = []
    for quality, keywords in CLAIM_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            matched.append(quality)

    if "confident" in normalized:
        matched.extend(["communication", "emotional_control"])
    if "social" in normalized:
        matched.extend(["teamwork", "communication"])
    if "air force" in normalized or "officer" in normalized:
        matched.extend(["responsibility", "discipline", "leadership"])
    if "dance" in normalized:
        matched.extend(["discipline", "communication", "resilience"])

    deduped: list[str] = []
    for quality in matched:
        if quality in QUALITY_LABELS and quality not in deduped:
            deduped.append(quality)
    return deduped[:4]


def audit_claim(claim: str, incidents: list[IncidentRecord]) -> dict[str, Any]:
    required = infer_claim_qualities(claim)
    normalized = normalize_text(claim)
    flags: list[str] = []
    if not required:
        flags.append("The claim is too broad. Tie it to a quality like responsibility, teamwork, discipline, or self-awareness.")
    if any(phrase in normalized for phrase in GENERIC_PHRASES):
        flags.append("This uses common interview language. It needs real proof.")
    if any(word in normalized for word in ("always", "never", "every situation")):
        flags.append("Absolute words are risky. Interviewers can challenge them easily.")

    evidence_rows: list[dict[str, Any]] = []
    for incident in incidents:
        if required:
            match_score = sum(incident.qualities.get(quality, 0.0) for quality in required) / len(required)
        else:
            match_score = max(incident.qualities.values(), default=0.0)
        evidence_score = match_score * 0.72 + (incident.strength / 100.0) * 0.28
        if evidence_score >= 0.25:
            evidence_rows.append(
                {
                    "incident_id": incident.id,
                    "text": incident.text,
                    "match": round(evidence_score * 100, 1),
                    "strength": incident.strength,
                    "status": incident.status,
                    "qualities": incident.qualities,
                }
            )

    evidence_rows.sort(key=lambda row: row["match"], reverse=True)
    best = evidence_rows[0]["match"] if evidence_rows else 0.0
    strong_count = sum(row["match"] >= 62 and row["strength"] >= 55 for row in evidence_rows)

    if best >= 72 and strong_count >= 2:
        verdict = "Supported"
    elif best >= 55 and strong_count >= 1:
        verdict = "Needs One More Proof"
    elif best >= 35:
        verdict = "Weak"
    else:
        verdict = "Unsupported"

    if flags and verdict == "Supported":
        verdict = "Supported With Risk"

    gaps = [QUALITY_DISPLAY[quality] for quality in required if not any(row["qualities"].get(quality, 0.0) >= 0.45 for row in evidence_rows)]
    if gaps:
        flags.append(f"Missing strong evidence for: {', '.join(gaps)}.")

    return {
        "claim": claim,
        "verdict": verdict,
        "required_qualities": required,
        "support_score": round(best, 1),
        "strong_evidence_count": strong_count,
        "flags": flags,
        "supporting_incidents": evidence_rows[:5],
        "crossfire_questions": generate_crossfire_questions(claim, required, evidence_rows[:2]),
        "repair_plan": claim_repair_plan(claim, required, evidence_rows[:3], flags),
    }


def claim_repair_plan(
    claim: str,
    qualities: list[str],
    evidence_rows: list[dict[str, Any]],
    flags: list[str],
) -> list[str]:
    plan: list[str] = []
    if not qualities:
        plan.append("Rewrite the claim around one clear quality instead of a broad personality statement.")
    if not evidence_rows:
        plan.append("Add one recent real incident before using this claim in an interview.")
    elif evidence_rows[0]["strength"] < 62:
        plan.append("Strengthen the best story with result and learning before using it.")
    if flags:
        plan.append("Remove absolute or generic wording. Use measured, verifiable language.")
    if len(evidence_rows) < 2:
        plan.append("Prepare a second example in case the interviewer asks for another proof.")
    if not plan:
        plan.append("Use the best story, then be ready to explain your exact role and mistake.")
    return plan[:4]


def generate_crossfire_questions(claim: str, qualities: list[str], supporting_incidents: list[dict[str, Any]]) -> list[str]:
    questions = [
        "What exactly did you do, not the group?",
        "What was difficult in that situation?",
        "What was the result and who benefited?",
        "What did you learn about yourself?",
        "What would the other person say about your role?",
    ]
    quality_questions = {
        "leadership": "Did people follow you because of authority, friendship, or clarity of action?",
        "teamwork": "Where did you adjust for the team instead of pushing only your view?",
        "communication": "How did you know the other person understood you?",
        "responsibility": "What consequence did you personally own?",
        "self_awareness": "What specific behavior did you change after this?",
        "emotional_control": "What did you feel internally, and how did you prevent it from affecting action?",
        "discipline": "For how many days did you maintain this without external pressure?",
        "resilience": "What was different in your second attempt?",
        "initiative": "Why did you act before being asked?",
        "planning": "What part of the plan failed, and how did you adjust?",
        "adaptability": "What did you change when the first plan stopped working?",
    }
    for quality in qualities:
        if quality in quality_questions:
            questions.append(quality_questions[quality])
    if not supporting_incidents:
        questions.append("Give one recent real incident that proves this claim.")
    else:
        questions.append("If the interviewer asks for another example, what will you say?")
    return questions[:7]


def analyze_practice_answer(answer: str) -> dict[str, Any]:
    blueprint = story_blueprint(answer)
    normalized = normalize_text(answer)
    flags: list[str] = []
    if len(answer.split()) < 35:
        flags.append("Answer is too short for a serious follow-up. Add one concrete detail.")
    if any(phrase in normalized for phrase in GENERIC_PHRASES):
        flags.append("Generic phrase detected. Replace it with what actually happened.")
    if "Personal Action" in blueprint["missing"]:
        flags.append("The interviewer may ask: what exactly did you do?")
    if "Result" in blueprint["missing"]:
        flags.append("Result is missing. Add the visible outcome.")
    if "Learning" in blueprint["missing"]:
        flags.append("Learning is missing. Add what changed after the incident.")
    if not flags:
        flags.append("This answer has a usable structure. Practice saying it in 45-60 seconds.")
    return {
        "score": blueprint["score"],
        "label": blueprint["label"],
        "checks": blueprint["checks"],
        "flags": flags,
    }


def evidence_map(incidents: list[IncidentRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for quality in QUALITY_LABELS:
        evidence_values = [incident.qualities.get(quality, 0.0) * incident.strength for incident in incidents]
        best = max(evidence_values, default=0.0)
        count = sum(value >= 28.0 for value in evidence_values)
        if best >= 62 and count >= 2:
            status = "Strong"
        elif best >= 42:
            status = "Developing"
        else:
            status = "Gap"
        rows.append(
            {
                "Quality": QUALITY_DISPLAY[quality],
                "Best Evidence": round(best, 1),
                "Usable Stories": int(count),
                "Status": status,
                "key": quality,
            }
        )
    return rows


def evidence_health(incidents: list[IncidentRecord], claims: list[Any] | None = None) -> dict[str, Any]:
    rows = evidence_map(incidents)
    strong = sum(row["Status"] == "Strong" for row in rows)
    developing = sum(row["Status"] == "Developing" for row in rows)
    gaps = sum(row["Status"] == "Gap" for row in rows)
    avg_strength = round(sum(incident.strength for incident in incidents) / max(len(incidents), 1), 1)
    coverage_score = int(round((strong * 100 + developing * 58) / max(len(rows), 1)))
    story_score = int(round(min(avg_strength, 100)))
    claim_count = len(claims or [])
    claim_bonus = min(claim_count * 4, 12)
    health_score = int(round(0.55 * coverage_score + 0.35 * story_score + claim_bonus))
    health_score = max(0, min(100, health_score))
    return {
        "health_score": health_score,
        "coverage_score": coverage_score,
        "story_score": story_score,
        "average_story_strength": avg_strength,
        "strong": strong,
        "developing": developing,
        "gaps": gaps,
    }


def piq_consistency_audit(piq: dict[str, str], incidents: list[IncidentRecord]) -> list[dict[str, Any]]:
    checks = [
        ("strengths", "Strengths"),
        ("weaknesses", "Weaknesses"),
        ("responsibilities", "Responsibilities"),
        ("achievements", "Achievements"),
        ("failures", "Failures"),
        ("hobbies", "Hobbies"),
    ]
    rows: list[dict[str, Any]] = []
    for field, label in checks:
        value = piq.get(field, "").strip()
        if not value:
            rows.append(
                {
                    "PIQ Area": label,
                    "Status": "Empty",
                    "Risk": "This area can trigger interview questions. Add truthful points.",
                    "Support": 0.0,
                }
            )
            continue
        audit = audit_claim(value, incidents) if incidents else None
        support = audit["support_score"] if audit else 0.0
        if support >= 65:
            status = "Backed"
            risk = "Has usable story support."
        elif support >= 38:
            status = "Thin"
            risk = "One or more claims need sharper incidents."
        else:
            status = "Risk"
            risk = "Likely to invite follow-up without strong proof."
        rows.append({"PIQ Area": label, "Status": status, "Risk": risk, "Support": support})
    return rows


def build_interview_brief(
    piq: dict[str, str],
    incidents: list[IncidentRecord],
    claims: list[Any],
) -> str:
    health = evidence_health(incidents, claims)
    rows = evidence_map(incidents)
    lines = [
        "# PIQ Crossfire Interview Brief",
        "",
        f"Evidence health: {health['health_score']}/100",
        f"Stories saved: {len(incidents)}",
        f"Claims audited: {len(claims)}",
        "",
        "## Strongest Evidence Areas",
    ]
    for row in sorted(rows, key=lambda item: item["Best Evidence"], reverse=True)[:5]:
        lines.append(f"- {row['Quality']}: {row['Status']} ({row['Best Evidence']}/100)")

    lines.extend(["", "## Best Stories"])
    for incident in sorted(incidents, key=lambda item: item.strength, reverse=True)[:5]:
        qualities = quality_summary(incident.qualities)
        lines.append(f"- {incident.text}")
        lines.append(f"  Evidence: {qualities}; strength {incident.strength:.1f}/100")

    if claims:
        lines.extend(["", "## Recent Claim Audits"])
        for claim in claims[:5]:
            lines.append(f"- {claim.text}: {claim.audit.get('verdict')} ({claim.audit.get('support_score')}/100)")

    consistency = piq_consistency_audit(piq, incidents)
    lines.extend(["", "## PIQ Risks"])
    for row in consistency:
        if row["Status"] in {"Empty", "Thin", "Risk"}:
            lines.append(f"- {row['PIQ Area']}: {row['Status']} - {row['Risk']}")

    lines.extend(["", "## Missions"])
    for mission in recommend_missions(incidents, piq):
        lines.append(f"- {mission['Quality']}: {mission['Mission']}")
    return "\n".join(lines)


def quality_summary(qualities: dict[str, float], limit: int = 4) -> str:
    sorted_items = sorted(qualities.items(), key=lambda item: item[1], reverse=True)[:limit]
    return ", ".join(f"{QUALITY_DISPLAY[key]} {value * 100:.0f}%" for key, value in sorted_items)


def recommend_missions(incidents: list[IncidentRecord], piq: dict[str, str] | None = None) -> list[dict[str, str]]:
    map_rows = evidence_map(incidents)
    weak_rows = [row for row in map_rows if row["Status"] == "Gap"]
    if not weak_rows:
        weak_rows = sorted(map_rows, key=lambda row: row["Best Evidence"])[:3]

    mission_bank = {
        "initiative": "Start one small pending task without waiting for instruction. Record what triggered your action.",
        "responsibility": "Own one task end-to-end today. Note the result and one mistake you corrected.",
        "teamwork": "Help one person complete something without taking over their work.",
        "communication": "Explain one current affairs topic to a friend in two minutes and ask what was unclear.",
        "leadership": "Coordinate a tiny group activity: roles, timeline, and result.",
        "planning": "Plan tomorrow in three blocks and review what actually happened.",
        "adaptability": "Change one failed plan today and document the alternate method.",
        "emotional_control": "When irritated or stressed, pause for 30 seconds, then choose the practical action.",
        "self_awareness": "Write one failure without blaming anyone. Add what you changed after it.",
        "discipline": "Protect one fixed practice slot today, even if it is only 20 minutes.",
        "resilience": "Retry one thing you recently avoided after a setback.",
    }

    missions: list[dict[str, str]] = []
    for row in weak_rows[:4]:
        key = row["key"]
        missions.append(
            {
                "Quality": row["Quality"],
                "Mission": mission_bank[key],
                "Why": f"Current evidence status: {row['Status']} with best proof {row['Best Evidence']}/100.",
                "Proof to Capture": "context -> your action -> result -> learning",
            }
        )
    return missions
