from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import PIQ_FIELDS, QUALITY_DISPLAY
from src.inference import (
    analyze_incident,
    analyze_practice_answer,
    audit_claim,
    build_interview_brief,
    ensure_bundle,
    evidence_health,
    evidence_map,
    extract_life_log_candidates,
    piq_consistency_audit,
    quality_summary,
    recommend_missions,
    split_incidents,
    story_blueprint,
    summarize_life_log_themes,
)
from src.store import (
    add_incident,
    clear_all,
    delete_incident,
    init_db,
    list_claims,
    list_incidents,
    load_piq,
    save_claim,
    save_piq,
)
from src.train import train_crossfire_model


st.set_page_config(page_title="PIQ Crossfire", layout="wide")

st.markdown(
    """
    <style>
    .block-container { max-width: 1220px; padding-top: 1.3rem; }
    div[data-testid="stMetric"] {
        border: 1px solid #e6e8eb;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        background: #ffffff;
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #101828 !important;
    }
    .note {
        border-left: 3px solid #2563eb;
        padding: 0.7rem 1rem;
        background: #f7faff;
        color: #1d2939;
        border-radius: 4px;
    }
    .risk {
        border-left: 3px solid #dc2626;
        padding: 0.7rem 1rem;
        background: #fff7f7;
        color: #1d2939;
        border-radius: 4px;
    }
    .hero {
        border: 1px solid #e6e8eb;
        border-radius: 8px;
        padding: 1rem 1.1rem;
        background: #f8fafc;
        margin-bottom: 1rem;
    }
    .card {
        border: 1px solid #e6e8eb;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        background: #ffffff;
        margin: 0.4rem 0;
    }
    .badge {
        display: inline-block;
        padding: 0.15rem 0.5rem;
        border-radius: 999px;
        background: #eff6ff;
        color: #1d4ed8;
        font-size: 0.78rem;
        margin-right: 0.25rem;
        margin-bottom: 0.25rem;
    }
    .quiet { color: #667085; font-size: 0.92rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_bundle() -> dict:
    return ensure_bundle()


def sample_incidents() -> list[str]:
    return [
        "During dance practice, two people disagreed about formations. I listened to both sides, adjusted the sequence, and helped the group finish the routine. I learned that leadership can be calm coordination, not loud control.",
        "During my internship, a teammate was stuck with a frontend bug. I recreated the issue, explained the state flow, and helped them fix it without taking over the work.",
        "While preparing for AFSB with a full workday, I missed my current affairs routine twice. I changed my schedule to a 20-minute morning slot and followed it for the next five days.",
        "When a trip plan with friends became confusing, I made a simple budget and route plan. The group used it and we avoided last-minute arguments.",
        "In a group discussion, I initially hesitated to speak. I noted the issue, practiced two-minute topic summaries, and spoke earlier in the next session.",
    ]


def badge_row(values: list[str]) -> None:
    html = "".join(f"<span class='badge'>{value}</span>" for value in values)
    st.markdown(html, unsafe_allow_html=True)


def render_blueprint(text: str) -> None:
    blueprint = story_blueprint(text)
    st.progress(blueprint["score"] / 100, text=f"{blueprint['label']} - {blueprint['score']}/100")
    checks = [f"{'Yes' if present else 'No'} {name}" for name, present in blueprint["checks"].items()]
    badge_row(checks)


def render_home() -> None:
    incidents = list_incidents()
    claims = list_claims()
    piq = load_piq()
    rows = evidence_map(incidents)
    health = evidence_health(incidents, claims)

    st.markdown(
        """
        <div class="hero">
        <b>PIQ Crossfire</b> checks one thing: whether your interview claims have real-life proof.
        It does not predict recommendation. It helps you enter AFSB with cleaner, sharper, more truthful examples.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Evidence Health", f"{health['health_score']}/100")
    col_b.metric("Stories", len(incidents), f"avg {health['average_story_strength']}/100")
    col_c.metric("Strong Areas", health["strong"])
    col_d.metric("Gaps", health["gaps"])

    st.progress(health["health_score"] / 100, text="Personal evidence health")

    left, right = st.columns([1.45, 1])
    with left:
        st.subheader("Evidence Map")
        map_frame = pd.DataFrame(rows).drop(columns=["key"])
        st.dataframe(map_frame, hide_index=True, use_container_width=True)
    with right:
        st.subheader("PIQ Risk Scan")
        consistency = pd.DataFrame(piq_consistency_audit(piq, incidents))
        st.dataframe(consistency, hide_index=True, use_container_width=True)

    if not incidents:
        st.markdown(
            "<div class='note'><b>Start:</b> import old logs or add 5-8 real incidents from internship, dance, friends, family, travel, failures, and preparation.</div>",
            unsafe_allow_html=True,
        )


def render_piq() -> None:
    st.subheader("Mini PIQ")
    st.caption("Use short truthful notes. The app will later test whether each area has evidence.")
    piq = load_piq()
    labels = {
        "hobbies": "Hobbies",
        "strengths": "Strengths",
        "weaknesses": "Weaknesses",
        "responsibilities": "Responsibilities",
        "achievements": "Achievements",
        "failures": "Failures",
        "friends_view": "What friends may say about you",
        "family_view": "What family may say about you",
        "daily_routine": "Daily routine",
    }
    updated: dict[str, str] = {}
    col_a, col_b = st.columns(2)
    for index, field in enumerate(PIQ_FIELDS):
        target = col_a if index % 2 == 0 else col_b
        with target:
            updated[field] = st.text_area(labels[field], value=piq.get(field, ""), height=92)
    if st.button("Save PIQ", type="primary", use_container_width=True):
        save_piq(updated)
        st.success("PIQ saved locally.")

    st.subheader("PIQ Consistency")
    incidents = list_incidents()
    consistency = pd.DataFrame(piq_consistency_audit(updated, incidents))
    st.dataframe(consistency, hide_index=True, use_container_width=True)


def eod_style_sample() -> str:
    return """Date: 10-06-2026
Continued PyTorch learning and understood the implementation of a Linear Regression model by analyzing the code flow line by line.
Studied torch.manual_seed for reproducibility and inference_mode for efficient model evaluation.
Explored loss functions including MAE and L1Loss, understanding their role in measuring model performance.
Learned about optimizers and training parameters including SGD and Adam.

Date: 12-06-2026
Continued PyTorch learning, focusing on techniques to improve neural network performance and model learning.
Implemented an improved neural network architecture for a regression problem, experimenting with network structure and training workflow.
Discussed project requirements, expected outcomes, and future implementation direction with Nikhil Sir.

Date: 18-05-2026
Worked on a contextual skincare catalog recommendation prototype using Python.
Built an intent extraction pipeline to identify contextual signals such as category, weather conditions, and time-based context from user queries.
Designed the recommendation flow to generate multiple personalized catalogs instead of static recommendations.
Added fallback handling mechanisms to avoid empty recommendations when exact matches are unavailable.

Date: 08-06-2026
Added the AI Intelligence Engine section within the Ansi sidebar navigation following existing architectural patterns.
Integrated PreferencePlus and BetterReply modules with route configuration and placeholder pages for future implementation.
Verified route registration, sidebar visibility, navigation flow, and overall integration without impacting existing resources.
"""


def render_import_logs(bundle: dict) -> None:
    st.subheader("Import Life Logs")
    st.caption("Paste EODs, notes, journals, or daily logs. The app extracts interview-worthy incidents and ranks them.")

    uploaded = st.file_uploader("Upload .txt or .md log", type=["txt", "md"])
    pasted = st.text_area(
        "Paste logs",
        height=230,
        placeholder="Paste EOD reports, daily logs, journal notes, or preparation notes here...",
    )

    col_a, col_b, col_c = st.columns([1.2, 1, 1])
    analyze_clicked = col_a.button("Extract interview moments", type="primary", use_container_width=True)
    sample_clicked = col_b.button("Use sample log", use_container_width=True)
    clear_clicked = col_c.button("Clear extraction", use_container_width=True)

    if clear_clicked:
        st.session_state.pop("log_candidates", None)
        st.success("Extraction cleared.")

    raw_parts: list[str] = []
    if uploaded is not None:
        raw_parts.append(uploaded.getvalue().decode("utf-8", errors="ignore"))
    if sample_clicked:
        raw_parts.append(eod_style_sample())
    if pasted.strip():
        raw_parts.append(pasted)

    if analyze_clicked or sample_clicked:
        raw_text = "\n\n".join(raw_parts)
        if not raw_text.strip():
            st.warning("Paste or upload logs first.")
        else:
            with st.spinner("Extracting interview-worthy moments..."):
                st.session_state["log_candidates"] = extract_life_log_candidates(raw_text, bundle, max_candidates=30, min_score=25.0)

    candidates = st.session_state.get("log_candidates", [])
    if not candidates:
        st.markdown(
            "<div class='note'><b>Tip:</b> old EODs are useful because they contain proof of learning, responsibility, problem solving, teamwork, and consistency.</div>",
            unsafe_allow_html=True,
        )
        return

    themes = summarize_life_log_themes(candidates)
    col_1, col_2, col_3 = st.columns(3)
    col_1.metric("Moments Found", len(candidates))
    col_2.metric("Themes", len(themes))
    col_3.metric("Best Signal", f"{max(candidate['import_score'] for candidate in candidates):.1f}/100")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Theme Summary")
        theme_frame = pd.DataFrame(themes).drop(columns=["key"]) if themes else pd.DataFrame()
        st.dataframe(theme_frame, hide_index=True, use_container_width=True)
    with right:
        st.subheader("Save to Story Bank")
        max_save = min(len(candidates), 12)
        if max_save == 1:
            save_count = 1
            st.caption("One moment is available to save.")
        else:
            save_count = st.slider("Top moments to save", min_value=1, max_value=max_save, value=min(max_save, 5))
        if st.button("Save top moments", type="primary", use_container_width=True):
            saved = 0
            for candidate in candidates[:save_count]:
                add_incident(candidate["text"], candidate["analysis"])
                saved += 1
            st.success(f"Saved {saved} extracted moment(s) to Story Bank.")

    st.subheader("Extracted Moments")
    rows = [
        {
            "Rank": index,
            "Import Score": candidate["import_score"],
            "Story Strength": candidate["strength"],
            "Status": candidate["status"],
            "Top Evidence": quality_summary(candidate["qualities"], limit=3),
            "Moment": candidate["text"][:140] + ("..." if len(candidate["text"]) > 140 else ""),
        }
        for index, candidate in enumerate(candidates, start=1)
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    for index, candidate in enumerate(candidates[:10], start=1):
        with st.expander(f"#{index} | {candidate['import_score']}/100 | {quality_summary(candidate['qualities'], limit=3)}"):
            st.write(candidate["text"])
            st.progress(candidate["blueprint"]["score"] / 100, text=f"{candidate['blueprint']['label']} - {candidate['blueprint']['score']}/100")
            checks = [f"{'Yes' if present else 'No'} {name}" for name, present in candidate["blueprint"]["checks"].items()]
            badge_row(checks)
            if candidate["risk_flags"]:
                st.markdown("<div class='risk'><b>Repair before using:</b></div>", unsafe_allow_html=True)
                for flag in candidate["risk_flags"]:
                    st.write(f"- {flag}")


def render_add_incidents(bundle: dict) -> None:
    st.subheader("Evidence Lab")
    st.caption("Paste one incident per line. Strong evidence has situation, pressure, personal action, result, and learning.")
    raw = st.text_area(
        "Incident dump",
        height=190,
        placeholder="Example: During dance practice, two people disagreed. I listened to both sides...",
    )
    col_a, col_b, col_c = st.columns([1.2, 1.2, 1])
    preview_clicked = col_a.button("Preview analysis", use_container_width=True)
    analyze_clicked = col_b.button("Analyze and save", type="primary", use_container_width=True)
    sample_clicked = col_c.button("Load sample stories", use_container_width=True)

    if sample_clicked:
        for text in sample_incidents():
            analysis = analyze_incident(text, bundle)
            add_incident(text, analysis)
        st.success("Sample stories added.")

    if analyze_clicked:
        incidents = split_incidents(raw)
        if not incidents:
            st.warning("Add at least one concrete incident.")
        else:
            for text in incidents:
                analysis = analyze_incident(text, bundle)
                add_incident(text, analysis)
            st.success(f"Saved {len(incidents)} incident(s).")

    if preview_clicked:
        incidents = split_incidents(raw)
        if not incidents:
            st.warning("Add at least one concrete incident.")
        for index, text in enumerate(incidents[:6], start=1):
            analysis = analyze_incident(text, bundle)
            with st.expander(f"Preview {index}: {analysis['status']} - {analysis['strength']}/100", expanded=True):
                st.write(text)
                st.caption(quality_summary(analysis["qualities"]))
                render_blueprint(text)
                for flag in analysis["risk_flags"]:
                    st.write(f"- {flag}")

    st.subheader("Story Bank")
    records = list_incidents()
    if not records:
        st.info("No incidents yet.")
        return

    for record in records[:20]:
        with st.expander(f"#{record.id} - {record.status} - {record.strength:.1f}/100", expanded=False):
            st.write(record.text)
            st.caption(quality_summary(record.qualities))
            render_blueprint(record.text)
            if st.button(f"Delete incident #{record.id}", key=f"delete-{record.id}"):
                delete_incident(record.id)
                st.rerun()


def render_claim_audit() -> None:
    st.subheader("Claim Audit")
    incidents = list_incidents()
    st.caption("Enter a claim you may say. The app checks whether your saved incidents can actually support it.")
    quick_claims = [
        "I am responsible.",
        "I can work well in a team.",
        "Dance has made me disciplined.",
        "I can stay calm under pressure.",
    ]
    selected_claim = st.selectbox("Quick claim starter", [""] + quick_claims)
    claim = st.text_input(
        "Enter a claim you may say in interview",
        value=selected_claim,
        placeholder="Example: I am responsible and can handle pressure.",
    )
    if st.button("Audit claim", type="primary", use_container_width=True):
        if not claim.strip():
            st.warning("Write a claim first.")
        elif not incidents:
            st.warning("Add incidents before auditing claims.")
        else:
            audit = audit_claim(claim, incidents)
            save_claim(claim, audit)
            st.session_state["latest_audit"] = audit

    audit = st.session_state.get("latest_audit")
    if audit:
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Verdict", audit["verdict"])
        col_b.metric("Support", f"{audit['support_score']:.1f}/100")
        col_c.metric("Strong Proofs", audit["strong_evidence_count"])

        qualities = [QUALITY_DISPLAY[key] for key in audit["required_qualities"]]
        badge_row(qualities if qualities else ["Needs clearer wording"])

        if audit["flags"]:
            st.markdown("<div class='risk'><b>Risks</b></div>", unsafe_allow_html=True)
            for flag in audit["flags"]:
                st.write(f"- {flag}")

        st.subheader("Repair Plan")
        for item in audit.get("repair_plan", []):
            st.write(f"- {item}")

        st.subheader("Supporting Incidents")
        if not audit["supporting_incidents"]:
            st.info("No strong supporting incident found.")
        for row in audit["supporting_incidents"]:
            with st.expander(f"Match {row['match']:.1f}% | Strength {row['strength']:.1f}/100 | {row['status']}", expanded=True):
                st.write(row["text"])
                render_blueprint(row["text"])


def render_crossfire() -> None:
    st.subheader("Crossfire Drill")
    latest = st.session_state.get("latest_audit")
    claims = list_claims()
    if latest:
        audit = latest
    elif claims:
        audit = claims[0].audit
    else:
        st.info("Audit a claim first.")
        return

    st.write(f"Claim: **{audit['claim']}**")
    for index, question in enumerate(audit["crossfire_questions"], start=1):
        with st.expander(f"Q{index}. {question}", expanded=index <= 2):
            answer = st.text_area("Practice answer", key=f"answer-{index}", height=105)
            st.caption("Use: situation -> pressure -> your action -> result -> learning.")
            check_clicked = st.button("Check answer", key=f"check-answer-{index}")
            feedback_key = f"answer-feedback-{index}"
            if check_clicked and answer.strip():
                st.session_state[feedback_key] = analyze_practice_answer(answer)
            if check_clicked and not answer.strip():
                st.warning("Write an answer first.")
            if feedback_key in st.session_state:
                feedback = st.session_state[feedback_key]
                st.progress(feedback["score"] / 100, text=f"{feedback['label']} - {feedback['score']}/100")
                checks = [f"{'Yes' if present else 'No'} {name}" for name, present in feedback["checks"].items()]
                badge_row(checks)
                for flag in feedback["flags"]:
                    st.write(f"- {flag}")


def render_missions() -> None:
    st.subheader("7-Day Evidence Missions")
    st.caption("These are not motivational tasks. Each mission is designed to create a real incident you can later explain.")
    incidents = list_incidents()
    piq = load_piq()
    missions = recommend_missions(incidents, piq)
    for day, mission in enumerate(missions, start=1):
        with st.container(border=True):
            st.write(f"**Day {day}: {mission['Quality']}**")
            st.write(mission["Mission"])
            st.caption(mission["Why"])
            st.code(mission["Proof to Capture"], language="text")


def render_brief() -> None:
    st.subheader("Interview Brief")
    piq = load_piq()
    incidents = list_incidents()
    claims = list_claims()
    brief = build_interview_brief(piq, incidents, claims)

    col_a, col_b = st.columns([1, 1])
    col_a.download_button(
        "Download brief",
        data=brief,
        file_name="piq_crossfire_interview_brief.md",
        mime="text/markdown",
        use_container_width=True,
    )
    col_b.metric("Brief Sections", brief.count("##"))

    st.markdown(brief)


def render_model(bundle: dict) -> None:
    st.subheader("Model")
    metrics = bundle.get("metrics", {})
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Quality F1", f"{metrics.get('quality_f1', 0) * 100:.1f}%")
    col_b.metric("Precision", f"{metrics.get('quality_precision', 0) * 100:.1f}%")
    col_c.metric("Status Accuracy", f"{metrics.get('status_accuracy', 0) * 100:.1f}%")
    col_d.metric("Strength MAE", f"{metrics.get('strength_mae', 0):.2f}")

    history = bundle.get("history", [])
    if history:
        history_frame = pd.DataFrame(history)
        st.line_chart(history_frame[["epoch", "loss", "train_loss"]].set_index("epoch"))


def main() -> None:
    init_db()
    st.title("PIQ Crossfire")
    st.caption("Claim-proof trainer for AFSB interview, PIQ, and self-description.")

    with st.sidebar:
        st.header("Controls")
        st.markdown(
            """
            **Workflow**
            1. Fill Mini PIQ
            2. Import logs or add incidents
            3. Audit claims
            4. Practice crossfire
            5. Download brief
            """
        )
        if st.button("Train fresh model", use_container_width=True):
            with st.spinner("Training PyTorch model..."):
                train_crossfire_model()
            st.cache_resource.clear()
            st.success("Model trained.")
        if st.button("Clear local data", use_container_width=True):
            clear_all()
            st.session_state.pop("latest_audit", None)
            st.success("Local PIQ, incidents, and claims cleared.")

    bundle = get_bundle()
    tabs = st.tabs(["Dashboard", "Mini PIQ", "Import Logs", "Evidence Lab", "Claim Audit", "Crossfire", "Missions", "Brief", "Model"])
    with tabs[0]:
        render_home()
    with tabs[1]:
        render_piq()
    with tabs[2]:
        render_import_logs(bundle)
    with tabs[3]:
        render_add_incidents(bundle)
    with tabs[4]:
        render_claim_audit()
    with tabs[5]:
        render_crossfire()
    with tabs[6]:
        render_missions()
    with tabs[7]:
        render_brief()
    with tabs[8]:
        render_model(bundle)


if __name__ == "__main__":
    main()
