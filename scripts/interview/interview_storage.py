"""
Interview Storage for Universal Interview Skill
Manages SQLite + Qdrant storage for interview initiatives and context
"""

import sqlite3
import json
import os
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path
from enum import Enum

# Qdrant client
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

# Sentence transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


class InitiativeStatus(Enum):
    """Status lifecycle for interview initiatives."""
    DRAFT = "draft"           # Interview in progress
    READY = "ready"           # Interview complete, available for ideation
    ACTIVE = "active"         # Currently being used by ideation
    ARCHIVED = "archived"     # Old, kept for reference


class InterviewDimension(Enum):
    """The 7 interview dimensions."""
    PROBLEM_SPACE = "problem_space"
    CONSTRAINTS = "constraints"
    ASSUMPTIONS = "assumptions"
    INTENT = "intent"
    PREFERENCES = "preferences"
    EXISTING_SOLUTIONS = "existing_solutions"
    RESOURCES = "resources"


class ConfidenceLevel(Enum):
    """Confidence levels for responses."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class ResponseSource(Enum):
    """Source of response data."""
    USER = "user"
    INJECTED = "injected"
    INFERRED = "inferred"


class InterviewStorage:
    """Manages persistent storage for interview initiatives and context."""

    INTERVIEW_COLLECTION = "interview_contexts"

    def __init__(
        self,
        db_path: str = None,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333
    ):
        # Set default database path
        if db_path is None:
            project_root = Path(__file__).parent.parent.parent
            db_path = str(project_root / "data" / "ideation.db")

        self.db_path = db_path
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port

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
        """Initialize SQLite database with interview tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Initiatives table - core initiative metadata
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS initiatives (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                original_domain TEXT,
                enriched_domain TEXT,
                status TEXT DEFAULT 'draft',
                template_scaffold TEXT,
                retention_days INTEGER DEFAULT 90,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                metadata JSON DEFAULT '{}'
            )
        ''')

        # Interview sessions table - individual interview conversations
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interview_sessions (
                id TEXT PRIMARY KEY,
                initiative_id TEXT NOT NULL,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ended_at TIMESTAMP,
                is_complete INTEGER DEFAULT 0,
                questions_asked INTEGER DEFAULT 0,
                dimensions_covered TEXT DEFAULT '[]',
                interruption_point TEXT,
                metadata JSON DEFAULT '{}',
                FOREIGN KEY (initiative_id) REFERENCES initiatives(id)
            )
        ''')

        # Interview responses table - individual Q&A pairs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interview_responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                initiative_id TEXT NOT NULL,
                dimension TEXT NOT NULL,
                question TEXT NOT NULL,
                response TEXT,
                confidence TEXT DEFAULT 'medium',
                source TEXT DEFAULT 'user',
                follow_up_depth INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSON DEFAULT '{}',
                FOREIGN KEY (session_id) REFERENCES interview_sessions(id),
                FOREIGN KEY (initiative_id) REFERENCES initiatives(id)
            )
        ''')

        # Gaps flagged table - knowledge gaps identified during interview
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interview_gaps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                initiative_id TEXT NOT NULL,
                dimension TEXT NOT NULL,
                gap_description TEXT NOT NULL,
                exploration_priority TEXT DEFAULT 'medium',
                resolved INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (initiative_id) REFERENCES initiatives(id)
            )
        ''')

        # Source attributions table - track what came from where
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS source_attributions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                initiative_id TEXT NOT NULL,
                dimension TEXT NOT NULL,
                source TEXT NOT NULL,
                citation TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (initiative_id) REFERENCES initiatives(id)
            )
        ''')

        # Interview analytics table (V2 - for learning loop)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interview_analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                initiative_id TEXT NOT NULL,
                metric_type TEXT NOT NULL,
                metric_value REAL,
                metadata JSON DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (initiative_id) REFERENCES initiatives(id)
            )
        ''')

        # Create indexes for common queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_initiatives_status ON initiatives(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_initiatives_updated ON initiatives(last_updated)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_initiative ON interview_sessions(initiative_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_responses_initiative ON interview_responses(initiative_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_responses_dimension ON interview_responses(dimension)')

        conn.commit()
        conn.close()

    def _init_qdrant(self):
        """Initialize Qdrant collection for semantic similarity."""
        if not self.qdrant:
            return

        collections = self.qdrant.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.INTERVIEW_COLLECTION not in collection_names:
            self.qdrant.create_collection(
                collection_name=self.INTERVIEW_COLLECTION,
                vectors_config=VectorParams(
                    size=384,  # all-MiniLM-L6-v2 dimension
                    distance=Distance.COSINE
                )
            )
            print(f"Created Qdrant collection: {self.INTERVIEW_COLLECTION}")

    # ===================
    # Initiative Methods
    # ===================

    def create_initiative(
        self,
        name: str,
        original_domain: str,
        template_scaffold: str = None,
        retention_days: int = 90,
        metadata: Dict = None
    ) -> str:
        """Create a new interview initiative."""
        initiative_id = str(uuid.uuid4())

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO initiatives (id, name, original_domain, template_scaffold, retention_days, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            initiative_id,
            name,
            original_domain,
            template_scaffold,
            retention_days,
            json.dumps(metadata or {})
        ))
        conn.commit()
        conn.close()

        # Store embedding for semantic similarity
        self._store_initiative_embedding(initiative_id, original_domain, name)

        return initiative_id

    def get_initiative(self, initiative_id: str) -> Optional[Dict]:
        """Get initiative by ID."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM initiatives WHERE id = ?', (initiative_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    def get_initiative_by_name(self, name: str) -> Optional[Dict]:
        """Get initiative by name."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM initiatives WHERE name = ?', (name,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    def update_initiative(
        self,
        initiative_id: str,
        **kwargs
    ) -> bool:
        """Update initiative fields."""
        allowed_fields = {'name', 'enriched_domain', 'status', 'template_scaffold',
                         'retention_days', 'completed_at', 'metadata'}
        update_fields = {k: v for k, v in kwargs.items() if k in allowed_fields}

        if not update_fields:
            return False

        # Always update last_updated
        update_fields['last_updated'] = datetime.now().isoformat()

        # Handle metadata as JSON
        if 'metadata' in update_fields:
            update_fields['metadata'] = json.dumps(update_fields['metadata'])

        set_clause = ', '.join(f'{k} = ?' for k in update_fields.keys())
        values = list(update_fields.values()) + [initiative_id]

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(f'UPDATE initiatives SET {set_clause} WHERE id = ?', values)
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()

        # Update embedding if domain changed
        if 'enriched_domain' in kwargs:
            initiative = self.get_initiative(initiative_id)
            if initiative:
                self._store_initiative_embedding(
                    initiative_id,
                    kwargs['enriched_domain'],
                    initiative['name']
                )

        return success

    def update_initiative_status(self, initiative_id: str, status: InitiativeStatus) -> bool:
        """Update initiative status."""
        return self.update_initiative(initiative_id, status=status.value)

    def list_initiatives(
        self,
        status: InitiativeStatus = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict]:
        """List initiatives with optional status filter."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if status:
            cursor.execute('''
                SELECT * FROM initiatives WHERE status = ?
                ORDER BY last_updated DESC LIMIT ? OFFSET ?
            ''', (status.value, limit, offset))
        else:
            cursor.execute('''
                SELECT * FROM initiatives
                ORDER BY last_updated DESC LIMIT ? OFFSET ?
            ''', (limit, offset))

        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def delete_initiative(self, initiative_id: str) -> bool:
        """Delete initiative and all associated data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Delete in order due to foreign keys
        cursor.execute('DELETE FROM source_attributions WHERE initiative_id = ?', (initiative_id,))
        cursor.execute('DELETE FROM interview_gaps WHERE initiative_id = ?', (initiative_id,))
        cursor.execute('DELETE FROM interview_analytics WHERE initiative_id = ?', (initiative_id,))
        cursor.execute('DELETE FROM interview_responses WHERE initiative_id = ?', (initiative_id,))
        cursor.execute('DELETE FROM interview_sessions WHERE initiative_id = ?', (initiative_id,))
        cursor.execute('DELETE FROM initiatives WHERE id = ?', (initiative_id,))

        success = cursor.rowcount > 0
        conn.commit()
        conn.close()

        # Delete from Qdrant
        if self.qdrant:
            try:
                self.qdrant.delete(
                    collection_name=self.INTERVIEW_COLLECTION,
                    points_selector=Filter(
                        must=[FieldCondition(key="initiative_id", match=MatchValue(value=initiative_id))]
                    )
                )
            except Exception as e:
                print(f"Warning: Could not delete from Qdrant: {e}")

        return success

    # ===================
    # Session Methods
    # ===================

    def create_session(self, initiative_id: str, metadata: Dict = None) -> str:
        """Create a new interview session for an initiative."""
        session_id = str(uuid.uuid4())

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO interview_sessions (id, initiative_id, metadata)
            VALUES (?, ?, ?)
        ''', (session_id, initiative_id, json.dumps(metadata or {})))
        conn.commit()
        conn.close()

        return session_id

    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session by ID."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM interview_sessions WHERE id = ?', (session_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    def get_latest_session(self, initiative_id: str) -> Optional[Dict]:
        """Get the most recent session for an initiative."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM interview_sessions
            WHERE initiative_id = ?
            ORDER BY started_at DESC LIMIT 1
        ''', (initiative_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    def update_session(self, session_id: str, **kwargs) -> bool:
        """Update session fields."""
        allowed_fields = {'ended_at', 'is_complete', 'questions_asked',
                         'dimensions_covered', 'interruption_point', 'metadata'}
        update_fields = {k: v for k, v in kwargs.items() if k in allowed_fields}

        if not update_fields:
            return False

        # Handle lists/dicts as JSON
        if 'dimensions_covered' in update_fields:
            update_fields['dimensions_covered'] = json.dumps(update_fields['dimensions_covered'])
        if 'metadata' in update_fields:
            update_fields['metadata'] = json.dumps(update_fields['metadata'])

        set_clause = ', '.join(f'{k} = ?' for k in update_fields.keys())
        values = list(update_fields.values()) + [session_id]

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(f'UPDATE interview_sessions SET {set_clause} WHERE id = ?', values)
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return success

    def get_incomplete_sessions(self, initiative_id: str) -> List[Dict]:
        """Get incomplete sessions for resume functionality."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM interview_sessions
            WHERE initiative_id = ? AND is_complete = 0
            ORDER BY started_at DESC
        ''', (initiative_id,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    # ===================
    # Response Methods
    # ===================

    def store_response(
        self,
        session_id: str,
        initiative_id: str,
        dimension: InterviewDimension,
        question: str,
        response: str,
        confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM,
        source: ResponseSource = ResponseSource.USER,
        follow_up_depth: int = 0,
        metadata: Dict = None
    ) -> int:
        """Store an interview response."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO interview_responses
            (session_id, initiative_id, dimension, question, response, confidence, source, follow_up_depth, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            session_id,
            initiative_id,
            dimension.value,
            question,
            response,
            confidence.value,
            source.value,
            follow_up_depth,
            json.dumps(metadata or {})
        ))
        response_id = cursor.lastrowid

        # Update session questions count
        cursor.execute('''
            UPDATE interview_sessions
            SET questions_asked = questions_asked + 1
            WHERE id = ?
        ''', (session_id,))

        conn.commit()
        conn.close()
        return response_id

    def get_responses_by_dimension(
        self,
        initiative_id: str,
        dimension: InterviewDimension
    ) -> List[Dict]:
        """Get all responses for a dimension."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM interview_responses
            WHERE initiative_id = ? AND dimension = ?
            ORDER BY created_at ASC
        ''', (initiative_id, dimension.value))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_all_responses(self, initiative_id: str) -> Dict[str, List[Dict]]:
        """Get all responses grouped by dimension."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM interview_responses
            WHERE initiative_id = ?
            ORDER BY dimension, created_at ASC
        ''', (initiative_id,))
        rows = cursor.fetchall()
        conn.close()

        # Group by dimension
        responses = {}
        for row in rows:
            dim = row['dimension']
            if dim not in responses:
                responses[dim] = []
            responses[dim].append(dict(row))
        return responses

    def get_dimensions_covered(self, initiative_id: str) -> List[str]:
        """Get list of dimensions that have responses."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT DISTINCT dimension FROM interview_responses
            WHERE initiative_id = ?
        ''', (initiative_id,))
        rows = cursor.fetchall()
        conn.close()
        return [row[0] for row in rows]

    # ===================
    # Gap Methods
    # ===================

    def flag_gap(
        self,
        initiative_id: str,
        dimension: InterviewDimension,
        gap_description: str,
        exploration_priority: str = "medium"
    ) -> int:
        """Flag a knowledge gap for exploration."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO interview_gaps (initiative_id, dimension, gap_description, exploration_priority)
            VALUES (?, ?, ?, ?)
        ''', (initiative_id, dimension.value, gap_description, exploration_priority))
        gap_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return gap_id

    def get_gaps(self, initiative_id: str, unresolved_only: bool = True) -> List[Dict]:
        """Get flagged gaps for an initiative."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if unresolved_only:
            cursor.execute('''
                SELECT * FROM interview_gaps
                WHERE initiative_id = ? AND resolved = 0
                ORDER BY exploration_priority DESC, created_at ASC
            ''', (initiative_id,))
        else:
            cursor.execute('''
                SELECT * FROM interview_gaps
                WHERE initiative_id = ?
                ORDER BY created_at ASC
            ''', (initiative_id,))

        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def resolve_gap(self, gap_id: int) -> bool:
        """Mark a gap as resolved."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('UPDATE interview_gaps SET resolved = 1 WHERE id = ?', (gap_id,))
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return success

    # ===================
    # Attribution Methods
    # ===================

    def add_attribution(
        self,
        initiative_id: str,
        dimension: InterviewDimension,
        source: ResponseSource,
        citation: str = None
    ) -> int:
        """Add a source attribution."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO source_attributions (initiative_id, dimension, source, citation)
            VALUES (?, ?, ?, ?)
        ''', (initiative_id, dimension.value, source.value, citation))
        attr_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return attr_id

    def get_attributions(self, initiative_id: str) -> Dict[str, List[Dict]]:
        """Get all attributions grouped by source."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM source_attributions
            WHERE initiative_id = ?
            ORDER BY source, dimension
        ''', (initiative_id,))
        rows = cursor.fetchall()
        conn.close()

        # Group by source
        attributions = {}
        for row in rows:
            src = row['source']
            if src not in attributions:
                attributions[src] = []
            attributions[src].append(dict(row))
        return attributions

    # ===================
    # Semantic Similarity
    # ===================

    def _store_initiative_embedding(self, initiative_id: str, domain: str, name: str):
        """Store initiative embedding in Qdrant."""
        if not self.qdrant or not self.embedder:
            return

        try:
            text = f"{name}. {domain}"
            embedding = self.embedder.encode(text).tolist()

            # Use hash of initiative_id as numeric ID for Qdrant
            numeric_id = abs(hash(initiative_id)) % (2**63)

            self.qdrant.upsert(
                collection_name=self.INTERVIEW_COLLECTION,
                points=[PointStruct(
                    id=numeric_id,
                    vector=embedding,
                    payload={
                        "initiative_id": initiative_id,
                        "name": name,
                        "domain": domain
                    }
                )]
            )
        except Exception as e:
            print(f"Warning: Could not store initiative embedding: {e}")

    def find_similar_initiatives(
        self,
        query: str,
        threshold: float = 0.7,
        limit: int = 5
    ) -> List[Dict]:
        """Find initiatives semantically similar to query."""
        if not self.qdrant or not self.embedder:
            return []

        try:
            embedding = self.embedder.encode(query).tolist()
            results = self.qdrant.search(
                collection_name=self.INTERVIEW_COLLECTION,
                query_vector=embedding,
                limit=limit
            )

            similar = []
            for r in results:
                if r.score >= threshold:
                    initiative = self.get_initiative(r.payload['initiative_id'])
                    if initiative:
                        similar.append({
                            'initiative': initiative,
                            'similarity': r.score
                        })
            return similar
        except Exception as e:
            print(f"Warning: Similarity search failed: {e}")
            return []

    # ===================
    # Context Building
    # ===================

    def build_context(self, initiative_id: str) -> Dict:
        """Build complete context object from initiative data."""
        initiative = self.get_initiative(initiative_id)
        if not initiative:
            return {}

        responses = self.get_all_responses(initiative_id)
        gaps = self.get_gaps(initiative_id)
        attributions = self.get_attributions(initiative_id)

        # Build dimension data with confidence and source
        dimensions = {}
        for dim in InterviewDimension:
            dim_responses = responses.get(dim.value, [])
            if dim_responses:
                # Combine responses for this dimension
                combined = ' '.join(r['response'] for r in dim_responses if r['response'])
                # Use most recent confidence
                confidence = dim_responses[-1]['confidence'] if dim_responses else 'unknown'
                # Determine primary source
                sources = [r['source'] for r in dim_responses]
                primary_source = max(set(sources), key=sources.count) if sources else 'unknown'

                dimensions[dim.value] = {
                    'response': combined,
                    'confidence': confidence,
                    'source': primary_source,
                    'response_count': len(dim_responses)
                }

        # Build gaps list
        gaps_flagged = [g['gap_description'] for g in gaps]

        # Build source attribution summary
        source_summary = {}
        for source, attrs in attributions.items():
            source_summary[source] = list(set(a['dimension'] for a in attrs))

        return {
            'initiative_id': initiative['id'],
            'initiative_name': initiative['name'],
            'status': initiative['status'],
            'created_at': initiative['created_at'],
            'last_updated': initiative['last_updated'],
            'original_domain': initiative['original_domain'],
            'enriched_domain': initiative['enriched_domain'],
            'template_scaffold': initiative['template_scaffold'],
            'dimensions': dimensions,
            'gaps_flagged': gaps_flagged,
            'source_attributions': source_summary
        }

    # ===================
    # Export
    # ===================

    def export_markdown(self, initiative_id: str) -> str:
        """Export initiative context as Markdown."""
        context = self.build_context(initiative_id)
        if not context:
            return ""

        lines = []
        lines.append(f"# Interview Context: {context['initiative_name']}")
        lines.append("")
        lines.append(f"**Initiative ID:** `{context['initiative_id']}`")
        lines.append(f"**Status:** {context['status']}")
        lines.append(f"**Created:** {context['created_at']}")
        lines.append(f"**Last Updated:** {context['last_updated']}")
        lines.append("")

        if context['template_scaffold']:
            lines.append(f"**Template Scaffold:** {context['template_scaffold']}")
            lines.append("")

        lines.append("## Original Domain")
        lines.append("")
        lines.append(context['original_domain'] or "_Not specified_")
        lines.append("")

        if context['enriched_domain']:
            lines.append("## Enriched Domain")
            lines.append("")
            lines.append(context['enriched_domain'])
            lines.append("")

        lines.append("## Interview Dimensions")
        lines.append("")

        for dim in InterviewDimension:
            dim_data = context['dimensions'].get(dim.value)
            dim_title = dim.value.replace('_', ' ').title()

            lines.append(f"### {dim_title}")
            lines.append("")

            if dim_data:
                lines.append(f"**Confidence:** {dim_data['confidence']} | **Source:** {dim_data['source']}")
                lines.append("")
                lines.append(dim_data['response'])
            else:
                lines.append("_Not covered in interview_")

            lines.append("")

        if context['gaps_flagged']:
            lines.append("## Knowledge Gaps (Flagged for Exploration)")
            lines.append("")
            for gap in context['gaps_flagged']:
                lines.append(f"- {gap}")
            lines.append("")

        if context['source_attributions']:
            lines.append("## Source Attributions")
            lines.append("")
            for source, dims in context['source_attributions'].items():
                lines.append(f"**{source.title()}:** {', '.join(dims)}")
            lines.append("")

        lines.append("---")
        lines.append(f"*Exported on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

        return '\n'.join(lines)

    def export_to_file(self, initiative_id: str, output_path: str = None) -> str:
        """Export initiative to Markdown file."""
        markdown = self.export_markdown(initiative_id)
        if not markdown:
            return ""

        if output_path is None:
            project_root = Path(__file__).parent.parent.parent
            output_dir = project_root / "output"
            output_dir.mkdir(exist_ok=True)
            initiative = self.get_initiative(initiative_id)
            name_slug = (initiative['name'] or 'initiative').replace(' ', '_').lower()
            output_path = str(output_dir / f"interview_{name_slug}_{initiative_id[:8]}.md")

        with open(output_path, 'w') as f:
            f.write(markdown)

        return output_path

    # ===================
    # Statistics
    # ===================

    def get_stats(self) -> Dict:
        """Get interview storage statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM initiatives")
        total_initiatives = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM initiatives WHERE status = 'draft'")
        draft_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM initiatives WHERE status = 'ready'")
        ready_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM interview_sessions")
        total_sessions = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM interview_responses")
        total_responses = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM interview_gaps WHERE resolved = 0")
        open_gaps = cursor.fetchone()[0]

        conn.close()

        # Qdrant stats
        qdrant_count = 0
        if self.qdrant:
            try:
                info = self.qdrant.get_collection(self.INTERVIEW_COLLECTION)
                qdrant_count = info.points_count
            except:
                pass

        return {
            'total_initiatives': total_initiatives,
            'draft_initiatives': draft_count,
            'ready_initiatives': ready_count,
            'total_sessions': total_sessions,
            'total_responses': total_responses,
            'open_gaps': open_gaps,
            'qdrant_vectors': qdrant_count
        }


# CLI interface
if __name__ == "__main__":
    import sys

    storage = InterviewStorage()

    if len(sys.argv) < 2:
        print("Usage: python interview_storage.py <command>")
        print("Commands: init, stats, list, export <initiative_id>")
        sys.exit(1)

    command = sys.argv[1]

    if command == "init":
        print("Interview storage initialized successfully!")
        print(f"SQLite: {storage.db_path}")

    elif command == "stats":
        stats = storage.get_stats()
        print(json.dumps(stats, indent=2))

    elif command == "list":
        initiatives = storage.list_initiatives()
        for init in initiatives:
            print(f"[{init['status']}] {init['id'][:8]}... - {init['name']}")

    elif command == "export" and len(sys.argv) > 2:
        initiative_id = sys.argv[2]
        path = storage.export_to_file(initiative_id)
        if path:
            print(f"Exported to: {path}")
        else:
            print(f"Initiative not found: {initiative_id}")

    else:
        print(f"Unknown command: {command}")
