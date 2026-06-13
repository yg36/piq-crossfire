from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import sqlite3
from pathlib import Path
from typing import Any

from .config import DB_PATH, PIQ_FIELDS


@dataclass(frozen=True)
class IncidentRecord:
    id: int
    text: str
    strength: float
    status: str
    qualities: dict[str, float]
    created_at: str


@dataclass(frozen=True)
class ClaimRecord:
    id: int
    text: str
    audit: dict[str, Any]
    created_at: str


def connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def init_db(db_path: Path = DB_PATH) -> None:
    with connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS piq (
                field TEXT PRIMARY KEY,
                value TEXT NOT NULL DEFAULT ''
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                strength REAL NOT NULL,
                status TEXT NOT NULL,
                qualities_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS claims (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                audit_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        for field in PIQ_FIELDS:
            connection.execute("INSERT OR IGNORE INTO piq(field, value) VALUES(?, '')", (field,))


def load_piq(db_path: Path = DB_PATH) -> dict[str, str]:
    init_db(db_path)
    with connect(db_path) as connection:
        rows = connection.execute("SELECT field, value FROM piq").fetchall()
    data = {row["field"]: row["value"] for row in rows}
    for field in PIQ_FIELDS:
        data.setdefault(field, "")
    return data


def save_piq(values: dict[str, str], db_path: Path = DB_PATH) -> None:
    init_db(db_path)
    with connect(db_path) as connection:
        for field in PIQ_FIELDS:
            connection.execute(
                "INSERT INTO piq(field, value) VALUES(?, ?) ON CONFLICT(field) DO UPDATE SET value=excluded.value",
                (field, values.get(field, "")),
            )


def add_incident(text: str, analysis: dict[str, Any], db_path: Path = DB_PATH) -> int:
    init_db(db_path)
    with connect(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO incidents(text, strength, status, qualities_json, created_at)
            VALUES(?, ?, ?, ?, ?)
            """,
            (
                text.strip(),
                float(analysis["strength"]),
                str(analysis["status"]),
                json.dumps(analysis["qualities"]),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        return int(cursor.lastrowid)


def list_incidents(db_path: Path = DB_PATH) -> list[IncidentRecord]:
    init_db(db_path)
    with connect(db_path) as connection:
        rows = connection.execute("SELECT * FROM incidents ORDER BY id DESC").fetchall()
    return [
        IncidentRecord(
            id=int(row["id"]),
            text=row["text"],
            strength=float(row["strength"]),
            status=row["status"],
            qualities=json.loads(row["qualities_json"]),
            created_at=row["created_at"],
        )
        for row in rows
    ]


def delete_incident(incident_id: int, db_path: Path = DB_PATH) -> None:
    init_db(db_path)
    with connect(db_path) as connection:
        connection.execute("DELETE FROM incidents WHERE id = ?", (incident_id,))


def save_claim(text: str, audit: dict[str, Any], db_path: Path = DB_PATH) -> int:
    init_db(db_path)
    with connect(db_path) as connection:
        cursor = connection.execute(
            "INSERT INTO claims(text, audit_json, created_at) VALUES(?, ?, ?)",
            (text.strip(), json.dumps(audit), datetime.now().isoformat(timespec="seconds")),
        )
        return int(cursor.lastrowid)


def list_claims(db_path: Path = DB_PATH) -> list[ClaimRecord]:
    init_db(db_path)
    with connect(db_path) as connection:
        rows = connection.execute("SELECT * FROM claims ORDER BY id DESC").fetchall()
    return [
        ClaimRecord(
            id=int(row["id"]),
            text=row["text"],
            audit=json.loads(row["audit_json"]),
            created_at=row["created_at"],
        )
        for row in rows
    ]


def clear_all(db_path: Path = DB_PATH) -> None:
    init_db(db_path)
    with connect(db_path) as connection:
        connection.execute("DELETE FROM incidents")
        connection.execute("DELETE FROM claims")
        for field in PIQ_FIELDS:
            connection.execute("UPDATE piq SET value='' WHERE field=?", (field,))

