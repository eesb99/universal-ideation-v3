"""
Memory Helper for Universal Ideation v3
Manages SQLite + Qdrant storage for ideas and learnings
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path

# Qdrant client
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    print("Warning: qdrant-client not installed. Vector storage disabled.")

# Sentence transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("Warning: sentence-transformers not installed. Embeddings disabled.")


class MemoryHelper:
    """Manages persistent storage for ideation sessions."""

    def __init__(
        self,
        db_path: str = None,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "universal_ideas"
    ):
        # Set default database path
        if db_path is None:
            project_root = Path(__file__).parent.parent.parent
            db_path = str(project_root / "data" / "ideation.db")

        self.db_path = db_path
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.collection_name = collection_name

        # Ensure data directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Initialize SQLite
        self._init_sqlite()

        # Initialize Qdrant
        self.qdrant = None
        if QDRANT_AVAILABLE:
            try:
                self.qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
                self._init_qdrant()
            except Exception as e:
                print(f"Warning: Could not connect to Qdrant: {e}")

        # Initialize embedding model
        self.embedder = None
        if EMBEDDINGS_AVAILABLE:
            try:
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                print(f"Warning: Could not load embedding model: {e}")

    def _init_sqlite(self):
        """Initialize SQLite database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Ideas table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ideas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                domain TEXT,
                mode TEXT,
                concept_name TEXT,
                description TEXT,
                target_audience TEXT,
                differentiation TEXT,
                scores JSON,
                weighted_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Learnings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learnings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                pattern TEXT,
                observation TEXT,
                action TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                domain TEXT,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                total_ideas INTEGER,
                best_score REAL,
                metadata JSON
            )
        ''')

        conn.commit()
        conn.close()

    def _init_qdrant(self):
        """Initialize Qdrant collection for semantic search."""
        if not self.qdrant:
            return

        # Check if collection exists
        collections = self.qdrant.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name not in collection_names:
            # Create collection with 384-dim vectors (all-MiniLM-L6-v2)
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=384,
                    distance=Distance.COSINE
                )
            )
            print(f"Created Qdrant collection: {self.collection_name}")

    def create_session(self, domain: str) -> str:
        """Create a new ideation session."""
        session_id = f"{domain.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO sessions (id, domain, started_at, total_ideas, best_score, metadata)
            VALUES (?, ?, ?, 0, 0, '{}')
        ''', (session_id, domain, datetime.now()))
        conn.commit()
        conn.close()

        return session_id

    def store_idea(
        self,
        session_id: str,
        domain: str,
        mode: str,
        concept_name: str,
        description: str,
        target_audience: str = "",
        differentiation: str = "",
        scores: Dict[str, float] = None,
        weighted_score: float = 0.0
    ) -> int:
        """Store an idea in SQLite and Qdrant."""

        # Store in SQLite
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO ideas (session_id, domain, mode, concept_name, description,
                             target_audience, differentiation, scores, weighted_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            session_id, domain, mode, concept_name, description,
            target_audience, differentiation,
            json.dumps(scores) if scores else '{}',
            weighted_score
        ))
        idea_id = cursor.lastrowid

        # Update session stats
        cursor.execute('''
            UPDATE sessions
            SET total_ideas = total_ideas + 1,
                best_score = MAX(best_score, ?)
            WHERE id = ?
        ''', (weighted_score, session_id))

        conn.commit()
        conn.close()

        # Store embedding in Qdrant
        if self.qdrant and self.embedder:
            try:
                # Create text for embedding
                text = f"{concept_name}. {description}"
                embedding = self.embedder.encode(text).tolist()

                self.qdrant.upsert(
                    collection_name=self.collection_name,
                    points=[PointStruct(
                        id=idea_id,
                        vector=embedding,
                        payload={
                            "session_id": session_id,
                            "domain": domain,
                            "mode": mode,
                            "concept_name": concept_name,
                            "weighted_score": weighted_score
                        }
                    )]
                )
            except Exception as e:
                print(f"Warning: Could not store embedding: {e}")

        return idea_id

    def store_learning(
        self,
        session_id: str,
        pattern: str,
        observation: str,
        action: str
    ) -> int:
        """Store a learning insight."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO learnings (session_id, pattern, observation, action)
            VALUES (?, ?, ?, ?)
        ''', (session_id, pattern, observation, action))
        learning_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return learning_id

    def get_ideas_by_session(self, session_id: str) -> List[Dict]:
        """Get all ideas for a session."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM ideas WHERE session_id = ? ORDER BY weighted_score DESC
        ''', (session_id,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_top_ideas(self, domain: str = None, limit: int = 10) -> List[Dict]:
        """Get top scoring ideas, optionally filtered by domain."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if domain:
            cursor.execute('''
                SELECT * FROM ideas WHERE domain LIKE ?
                ORDER BY weighted_score DESC LIMIT ?
            ''', (f"%{domain}%", limit))
        else:
            cursor.execute('''
                SELECT * FROM ideas ORDER BY weighted_score DESC LIMIT ?
            ''', (limit,))

        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def check_novelty(self, text: str, threshold: float = 0.7) -> Dict:
        """Check if idea is novel by comparing to existing embeddings."""
        if not self.qdrant or not self.embedder:
            return {"is_novel": True, "similarity": 0.0, "similar_ideas": []}

        try:
            embedding = self.embedder.encode(text).tolist()
            results = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=embedding,
                limit=5
            )

            if not results:
                return {"is_novel": True, "similarity": 0.0, "similar_ideas": []}

            top_similarity = results[0].score
            similar_ideas = [
                {
                    "concept_name": r.payload.get("concept_name", "Unknown"),
                    "similarity": r.score
                }
                for r in results if r.score > 0.5
            ]

            return {
                "is_novel": top_similarity < threshold,
                "similarity": top_similarity,
                "similar_ideas": similar_ideas
            }
        except Exception as e:
            print(f"Warning: Novelty check failed: {e}")
            return {"is_novel": True, "similarity": 0.0, "similar_ideas": []}

    def get_stats(self) -> Dict:
        """Get database statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM ideas")
        total_ideas = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM sessions")
        total_sessions = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM learnings")
        total_learnings = cursor.fetchone()[0]

        cursor.execute("SELECT MAX(weighted_score) FROM ideas")
        best_score = cursor.fetchone()[0] or 0

        cursor.execute("SELECT AVG(weighted_score) FROM ideas")
        avg_score = cursor.fetchone()[0] or 0

        conn.close()

        # Get Qdrant stats
        qdrant_count = 0
        if self.qdrant:
            try:
                info = self.qdrant.get_collection(self.collection_name)
                qdrant_count = info.points_count
            except:
                pass

        return {
            "total_ideas": total_ideas,
            "total_sessions": total_sessions,
            "total_learnings": total_learnings,
            "best_score": best_score,
            "avg_score": round(avg_score, 2),
            "qdrant_vectors": qdrant_count
        }

    def export_session(self, session_id: str, output_path: str = None) -> str:
        """Export session to JSON file."""
        ideas = self.get_ideas_by_session(session_id)

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        row = cursor.fetchone()
        session = dict(row) if row else {}

        cursor.execute("SELECT * FROM learnings WHERE session_id = ?", (session_id,))
        learnings = [dict(row) for row in cursor.fetchall()]

        conn.close()

        export_data = {
            "session": session,
            "ideas": ideas,
            "learnings": learnings,
            "exported_at": datetime.now().isoformat()
        }

        if output_path is None:
            project_root = Path(__file__).parent.parent.parent
            output_dir = project_root / "output"
            output_dir.mkdir(exist_ok=True)
            output_path = str(output_dir / f"{session_id}.json")

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

        return output_path


# CLI interface
if __name__ == "__main__":
    import sys

    helper = MemoryHelper()

    if len(sys.argv) < 2:
        print("Usage: python memory_helper.py <command>")
        print("Commands: init, status, stats, export <session_id>")
        sys.exit(1)

    command = sys.argv[1]

    if command == "init":
        print("Database initialized successfully!")
        print(f"SQLite: {helper.db_path}")
        print(f"Qdrant: {helper.qdrant_host}:{helper.qdrant_port}")

    elif command == "status":
        stats = helper.get_stats()
        print(f"Database Status:")
        print(f"  SQLite path: {helper.db_path}")
        print(f"  Total ideas: {stats['total_ideas']}")
        print(f"  Total sessions: {stats['total_sessions']}")
        print(f"  Total learnings: {stats['total_learnings']}")
        print(f"  Best score: {stats['best_score']}")
        print(f"  Avg score: {stats['avg_score']}")
        print(f"  Qdrant vectors: {stats['qdrant_vectors']}")

    elif command == "stats":
        stats = helper.get_stats()
        print(json.dumps(stats, indent=2))

    elif command == "export" and len(sys.argv) > 2:
        session_id = sys.argv[2]
        path = helper.export_session(session_id)
        print(f"Exported to: {path}")

    else:
        print(f"Unknown command: {command}")
