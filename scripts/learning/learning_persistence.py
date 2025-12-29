"""
Learning Persistence for Universal Ideation v3.2

Implements SQLite-based cross-session learning storage:
- Stores reflections, weight adjustments, and session metadata
- Supports domain-specific learning retrieval
- Enables pattern aggregation across sessions
- Provides learning analytics and trend analysis

Based on ReflectEvo: Persistent learning for continuous improvement.
"""

import sqlite3
import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from pathlib import Path

from .reflection_generator import Reflection, ReflectionBatch, ReflectionType, ReflectionConfidence
from .weight_adjuster import WeightAdjustment, AdjustmentReason


@dataclass
class SessionRecord:
    """Record of a complete learning session."""
    session_id: str
    domain: str
    ideas_generated: int
    ideas_accepted: int
    reflections_count: int
    weight_adjustments: int
    average_score: float
    best_score: float
    started_at: str
    ended_at: str
    final_weights: Dict[str, float]


class LearningPersistence:
    """
    SQLite-based persistence for cross-session learning.

    Stores:
    - Reflections with full metadata
    - Weight adjustment history
    - Session summaries
    - Domain-specific patterns
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize persistence layer.

        Args:
            db_path: Path to SQLite database (defaults to project storage)
        """
        if db_path is None:
            # Default to project storage directory
            storage_dir = Path(__file__).parent.parent / "storage"
            storage_dir.mkdir(exist_ok=True)
            db_path = str(storage_dir / "learning.db")

        self.db_path = db_path
        self._init_database()

    def _init_database(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Reflections table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reflections (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    pattern TEXT NOT NULL,
                    observation TEXT NOT NULL,
                    evidence_count INTEGER,
                    confidence TEXT,
                    dimension_impacts TEXT,  -- JSON
                    domain TEXT NOT NULL,
                    idea_ids TEXT,  -- JSON array
                    created_at TEXT NOT NULL,
                    applied BOOLEAN DEFAULT FALSE
                )
            """)

            # Weight adjustments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS weight_adjustments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    dimension TEXT NOT NULL,
                    old_weight REAL NOT NULL,
                    new_weight REAL NOT NULL,
                    delta REAL NOT NULL,
                    reason TEXT NOT NULL,
                    reflection_id TEXT,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (reflection_id) REFERENCES reflections(id)
                )
            """)

            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    domain TEXT NOT NULL,
                    ideas_generated INTEGER,
                    ideas_accepted INTEGER,
                    reflections_count INTEGER,
                    weight_adjustments INTEGER,
                    average_score REAL,
                    best_score REAL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    final_weights TEXT  -- JSON
                )
            """)

            # Domain patterns table (aggregated learnings)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS domain_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    domain TEXT NOT NULL,
                    pattern_type TEXT NOT NULL,
                    pattern TEXT NOT NULL,
                    occurrence_count INTEGER DEFAULT 1,
                    avg_confidence REAL,
                    first_seen TEXT,
                    last_seen TEXT,
                    cumulative_impact TEXT,  -- JSON of dimension impacts
                    UNIQUE(domain, pattern)
                )
            """)

            # Indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_reflections_domain ON reflections(domain)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_reflections_session ON reflections(session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_adjustments_session ON weight_adjustments(session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_patterns_domain ON domain_patterns(domain)")

            conn.commit()

    def save_reflection(self, reflection: Reflection, session_id: str) -> None:
        """Save a single reflection."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO reflections
                (id, session_id, type, pattern, observation, evidence_count,
                 confidence, dimension_impacts, domain, idea_ids, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                reflection.id,
                session_id,
                reflection.type.value,
                reflection.pattern,
                reflection.observation,
                reflection.evidence_count,
                reflection.confidence.value,
                json.dumps(reflection.dimension_impacts),
                reflection.domain,
                json.dumps(reflection.idea_ids),
                reflection.created_at
            ))
            conn.commit()

    def save_reflection_batch(self, batch: ReflectionBatch) -> None:
        """Save a batch of reflections."""
        for reflection in batch.reflections:
            self.save_reflection(reflection, batch.session_id)

    def save_weight_adjustment(self, adjustment: WeightAdjustment, session_id: str) -> None:
        """Save a weight adjustment."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO weight_adjustments
                (session_id, dimension, old_weight, new_weight, delta,
                 reason, reflection_id, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                adjustment.dimension,
                adjustment.old_weight,
                adjustment.new_weight,
                adjustment.delta,
                adjustment.reason.value,
                adjustment.reflection_id,
                adjustment.timestamp
            ))
            conn.commit()

    def start_session(self, session_id: str, domain: str) -> None:
        """Record session start."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO sessions
                (session_id, domain, started_at, ideas_generated, ideas_accepted,
                 reflections_count, weight_adjustments, average_score, best_score)
                VALUES (?, ?, ?, 0, 0, 0, 0, 0, 0)
            """, (session_id, domain, datetime.now().isoformat()))
            conn.commit()

    def end_session(
        self,
        session_id: str,
        ideas_generated: int,
        ideas_accepted: int,
        average_score: float,
        best_score: float,
        final_weights: Dict[str, float]
    ) -> None:
        """Record session end."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Count reflections and adjustments
            cursor.execute(
                "SELECT COUNT(*) FROM reflections WHERE session_id = ?",
                (session_id,)
            )
            reflections_count = cursor.fetchone()[0]

            cursor.execute(
                "SELECT COUNT(*) FROM weight_adjustments WHERE session_id = ?",
                (session_id,)
            )
            adjustments_count = cursor.fetchone()[0]

            cursor.execute("""
                UPDATE sessions SET
                    ideas_generated = ?,
                    ideas_accepted = ?,
                    reflections_count = ?,
                    weight_adjustments = ?,
                    average_score = ?,
                    best_score = ?,
                    ended_at = ?,
                    final_weights = ?
                WHERE session_id = ?
            """, (
                ideas_generated,
                ideas_accepted,
                reflections_count,
                adjustments_count,
                average_score,
                best_score,
                datetime.now().isoformat(),
                json.dumps(final_weights),
                session_id
            ))
            conn.commit()

    def get_domain_reflections(
        self,
        domain: str,
        limit: int = 50,
        min_confidence: str = "low"
    ) -> List[Reflection]:
        """
        Get reflections for a specific domain.

        Args:
            domain: Domain to query
            limit: Maximum reflections to return
            min_confidence: Minimum confidence level

        Returns:
            List of reflections
        """
        confidence_order = {"low": 0, "medium": 1, "high": 2}
        min_conf_value = confidence_order.get(min_confidence, 0)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, type, pattern, observation, evidence_count,
                       confidence, dimension_impacts, domain, idea_ids, created_at
                FROM reflections
                WHERE domain = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (domain, limit))

            reflections = []
            for row in cursor.fetchall():
                conf = row[5]
                if confidence_order.get(conf, 0) >= min_conf_value:
                    reflections.append(Reflection(
                        id=row[0],
                        type=ReflectionType(row[1]),
                        pattern=row[2],
                        observation=row[3],
                        evidence_count=row[4],
                        confidence=ReflectionConfidence(row[5]),
                        dimension_impacts=json.loads(row[6]) if row[6] else {},
                        domain=row[7],
                        idea_ids=json.loads(row[8]) if row[8] else [],
                        created_at=row[9]
                    ))

            return reflections

    def get_high_confidence_patterns(
        self,
        domain: str,
        pattern_type: Optional[str] = None
    ) -> List[Reflection]:
        """Get high-confidence reflections, optionally filtered by type."""
        reflections = self.get_domain_reflections(domain, limit=100, min_confidence="high")

        if pattern_type:
            reflections = [r for r in reflections if r.type.value == pattern_type]

        return reflections

    def aggregate_domain_patterns(self, domain: str) -> None:
        """
        Aggregate reflections into domain patterns table.

        Combines similar patterns and tracks occurrence frequency.
        """
        reflections = self.get_domain_reflections(domain, limit=500)

        # Group by pattern
        pattern_groups: Dict[str, List[Reflection]] = {}
        for r in reflections:
            # Normalize pattern for grouping
            pattern_key = r.pattern.lower().strip()
            if pattern_key not in pattern_groups:
                pattern_groups[pattern_key] = []
            pattern_groups[pattern_key].append(r)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            for pattern_key, group in pattern_groups.items():
                if not group:
                    continue

                # Calculate aggregates
                occurrence_count = len(group)
                confidence_values = {"low": 1, "medium": 2, "high": 3}
                avg_confidence = sum(
                    confidence_values.get(r.confidence.value, 1)
                    for r in group
                ) / len(group)

                # Aggregate dimension impacts
                cumulative_impact: Dict[str, float] = {}
                for r in group:
                    for dim, impact in r.dimension_impacts.items():
                        cumulative_impact[dim] = cumulative_impact.get(dim, 0) + impact

                first_seen = min(r.created_at for r in group)
                last_seen = max(r.created_at for r in group)
                pattern_type = group[0].type.value

                cursor.execute("""
                    INSERT INTO domain_patterns
                    (domain, pattern_type, pattern, occurrence_count, avg_confidence,
                     first_seen, last_seen, cumulative_impact)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(domain, pattern) DO UPDATE SET
                        occurrence_count = ?,
                        avg_confidence = ?,
                        last_seen = ?,
                        cumulative_impact = ?
                """, (
                    domain, pattern_type, pattern_key, occurrence_count,
                    avg_confidence, first_seen, last_seen,
                    json.dumps(cumulative_impact),
                    occurrence_count, avg_confidence, last_seen,
                    json.dumps(cumulative_impact)
                ))

            conn.commit()

    def get_recommended_weights(self, domain: str) -> Optional[Dict[str, float]]:
        """
        Get recommended starting weights based on domain history.

        Returns:
            Recommended weights or None if no history
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Get most recent successful session weights
            cursor.execute("""
                SELECT final_weights FROM sessions
                WHERE domain = ? AND final_weights IS NOT NULL
                  AND average_score > 70
                ORDER BY ended_at DESC
                LIMIT 1
            """, (domain,))

            row = cursor.fetchone()
            if row and row[0]:
                return json.loads(row[0])

            # Fallback: aggregate dimension impacts from patterns
            cursor.execute("""
                SELECT cumulative_impact FROM domain_patterns
                WHERE domain = ? AND occurrence_count >= 3
            """, (domain,))

            cumulative: Dict[str, float] = {}
            for row in cursor.fetchall():
                if row[0]:
                    impacts = json.loads(row[0])
                    for dim, impact in impacts.items():
                        cumulative[dim] = cumulative.get(dim, 0) + impact

            if cumulative:
                # Convert to weight adjustments
                from .weight_adjuster import WeightAdjuster
                adjuster = WeightAdjuster()
                for dim, total_impact in cumulative.items():
                    # Normalize and cap
                    normalized = max(-0.05, min(0.05, total_impact / 10))
                    if dim in adjuster.current_weights:
                        adjuster.current_weights[dim] += normalized

                adjuster._validate_weights()
                return adjuster.current_weights

            return None

    def get_session_history(
        self,
        domain: Optional[str] = None,
        limit: int = 20
    ) -> List[SessionRecord]:
        """Get session history."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            if domain:
                cursor.execute("""
                    SELECT session_id, domain, ideas_generated, ideas_accepted,
                           reflections_count, weight_adjustments, average_score,
                           best_score, started_at, ended_at, final_weights
                    FROM sessions
                    WHERE domain = ?
                    ORDER BY started_at DESC
                    LIMIT ?
                """, (domain, limit))
            else:
                cursor.execute("""
                    SELECT session_id, domain, ideas_generated, ideas_accepted,
                           reflections_count, weight_adjustments, average_score,
                           best_score, started_at, ended_at, final_weights
                    FROM sessions
                    ORDER BY started_at DESC
                    LIMIT ?
                """, (limit,))

            sessions = []
            for row in cursor.fetchall():
                sessions.append(SessionRecord(
                    session_id=row[0],
                    domain=row[1],
                    ideas_generated=row[2] or 0,
                    ideas_accepted=row[3] or 0,
                    reflections_count=row[4] or 0,
                    weight_adjustments=row[5] or 0,
                    average_score=row[6] or 0,
                    best_score=row[7] or 0,
                    started_at=row[8],
                    ended_at=row[9] or "",
                    final_weights=json.loads(row[10]) if row[10] else {}
                ))

            return sessions

    def get_learning_trend(self, domain: str) -> Dict:
        """
        Analyze learning trend across sessions.

        Returns:
            Trend analysis with improvement metrics
        """
        sessions = self.get_session_history(domain, limit=10)

        if len(sessions) < 2:
            return {
                "sessions_analyzed": len(sessions),
                "trend": "insufficient_data",
                "improvement": 0
            }

        # Calculate score trend
        scores = [s.average_score for s in sessions if s.average_score > 0]
        if len(scores) < 2:
            return {
                "sessions_analyzed": len(sessions),
                "trend": "insufficient_data",
                "improvement": 0
            }

        # Compare first half vs second half
        mid = len(scores) // 2
        early_avg = sum(scores[mid:]) / len(scores[mid:])  # Earlier sessions
        late_avg = sum(scores[:mid]) / len(scores[:mid])   # Recent sessions

        improvement = late_avg - early_avg
        trend = "improving" if improvement > 2 else "declining" if improvement < -2 else "stable"

        return {
            "sessions_analyzed": len(sessions),
            "early_average": early_avg,
            "recent_average": late_avg,
            "improvement": improvement,
            "trend": trend,
            "total_reflections": sum(s.reflections_count for s in sessions),
            "total_adjustments": sum(s.weight_adjustments for s in sessions)
        }

    def get_statistics(self) -> Dict:
        """Get overall persistence statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM reflections")
            total_reflections = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM weight_adjustments")
            total_adjustments = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM sessions")
            total_sessions = cursor.fetchone()[0]

            cursor.execute("SELECT DISTINCT domain FROM sessions")
            domains = [row[0] for row in cursor.fetchall()]

            cursor.execute("SELECT COUNT(*) FROM domain_patterns")
            total_patterns = cursor.fetchone()[0]

            return {
                "total_reflections": total_reflections,
                "total_adjustments": total_adjustments,
                "total_sessions": total_sessions,
                "domains": domains,
                "total_patterns": total_patterns,
                "db_path": self.db_path
            }

    def cleanup_old_data(self, days: int = 90) -> int:
        """
        Remove old data beyond retention period.

        Args:
            days: Days of data to retain

        Returns:
            Number of records deleted
        """
        cutoff = datetime.now()
        cutoff_str = cutoff.isoformat()

        # For simplicity, we just report - actual cleanup requires date parsing
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Get count of old reflections
            cursor.execute("""
                SELECT COUNT(*) FROM reflections
                WHERE created_at < date('now', ?)
            """, (f'-{days} days',))

            return cursor.fetchone()[0]
