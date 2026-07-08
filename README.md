# PIQ Crossfire

PIQ Crossfire is a local Streamlit + PyTorch app for AFSB/SSB interview preparation. It does not predict selection, score personality, or pretend to replace the SSB process. It helps you check whether your PIQ, self-description, and interview claims are backed by real incidents from your own life.

## Core Idea

Most aspirants prepare claims:

```text
I am responsible.
I can work in a team.
I stay calm under pressure.
Dance made me disciplined.
```

PIQ Crossfire asks the harder question:

```text
Can you prove this with a real incident?
```

The app turns life logs, EODs, journal notes, and direct incident entries into an evidence bank. It then audits claims, generates crossfire questions, and exports an interview brief.

## Features

- Mini PIQ storage with SQLite
- EOD, journal, and daily-log import
- Interview-worthy moment extraction from messy logs
- Evidence theme summary across imported logs
- Incident tagging for qualities such as responsibility, teamwork, communication, discipline, resilience, and self-awareness
- Story anatomy check: situation, pressure, personal action, people context, result, and learning
- Claim audit against saved incidents
- Repair plan for weak, vague, risky, or unsupported claims
- Crossfire follow-up questions from your own claims
- Practice-answer feedback using SARL-style structure
- Seven-day evidence missions for weak areas
- Downloadable Markdown interview brief

## Tech Stack

- Streamlit for the UI
- PyTorch for the multi-task text model
- SQLite for local personal data
- NumPy and pandas for feature/data handling
- Custom hashed n-gram text features
- Rule-based evidence consistency layer
- pytest for smoke tests

## Project Structure

```text
piq-crossfire/
  app.py                     Streamlit app
  requirements.txt           Python dependencies
  scripts/train_model.py     Manual training entry point
  src/config.py              Labels, paths, app config
  src/inference.py           Analysis, extraction, audits, missions
  src/model.py               PyTorch model
  src/store.py               SQLite persistence
  src/synthetic_data.py      Synthetic training dataset
  src/text_features.py       Text vectorization and story signals
  src/train.py               Training loop and model artifact saving
  tests/test_pipeline.py     Backend smoke tests
```

## Quick Start

```powershell
cd E:\YG\codes\projects\piq-crossfire
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\train_model.py
streamlit run app.py
```

The app also trains automatically on first launch if no model artifact exists.

## Recommended Workflow

1. Fill **Mini PIQ** with truthful short notes.
2. Use **Import Logs** to paste EODs, journals, or daily prep notes.
3. Save the best extracted moments into **Story Bank**.
4. Use **Claim Audit** for lines you may say in interview.
5. Practice follow-up questions in **Crossfire**.
6. Review **Missions** to create missing evidence.
7. Export the **Brief** before revision.

## Model Tasks

The PyTorch backend uses one shared encoder with multiple heads:

- Multi-label quality tagging with `BCEWithLogitsLoss`
- Story strength regression with `SmoothL1Loss`
- Evidence status classification with `CrossEntropyLoss`

The model is intentionally combined with deterministic rules. For this type of preparation, transparent evidence checks are more useful than opaque scoring.

## Privacy Notes

- Personal PIQ data and incidents are stored locally in SQLite.
- Local database files are ignored by Git.
- Model artifacts are ignored by Git and can be regenerated.
- Do not commit private journal entries, real PIQ details, or sensitive personal data.

## Tests

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest -q
```

## Important Disclaimer

PIQ Crossfire is a preparation and self-reflection tool. It is not affiliated with the Indian Armed Forces, AFSB, SSB, or any official selection board. It should not be used as a selection predictor.

<!-- recruiter-review:start -->
## Review Status

Reviewed for recruiter visibility, setup clarity, and AI/ML positioning on June 13, 2026.
<!-- recruiter-review:end -->

<!-- repository-refresh: 2026-06-29 | preserved-order-rank: 011/71 -->
