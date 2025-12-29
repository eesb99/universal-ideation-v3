#!/usr/bin/env python3
"""
Backup script for Universal Ideation v3.2
Backs up SQLite database, Qdrant vectors, and exports to JSON
"""

import os
import shutil
import sqlite3
import json
import requests
from datetime import datetime
from pathlib import Path

# Qdrant client
try:
    from qdrant_client import QdrantClient
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

# Paths
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
BACKUP_DIR = PROJECT_ROOT / "backups"
DB_PATH = DATA_DIR / "ideation.db"

# Qdrant config
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION = "universal_ideas"


def get_qdrant_client():
    """Get Qdrant client if available."""
    if not QDRANT_AVAILABLE:
        return None
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        # Test connection
        client.get_collections()
        return client
    except Exception:
        return None


def get_qdrant_stats():
    """Get Qdrant collection statistics."""
    client = get_qdrant_client()
    if not client:
        return None

    try:
        info = client.get_collection(QDRANT_COLLECTION)
        return {
            "vectors_count": info.points_count,
            "status": info.status.value if hasattr(info.status, 'value') else str(info.status),
            "vector_size": info.config.params.vectors.size if info.config.params.vectors else None
        }
    except Exception as e:
        return {"error": str(e)}


def backup_qdrant(backup_name: str):
    """Create Qdrant collection snapshot."""
    client = get_qdrant_client()
    if not client:
        print("Qdrant not available, skipping vector backup.")
        return None

    try:
        # Create snapshot via REST API (more reliable for file download)
        url = f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections/{QDRANT_COLLECTION}/snapshots"
        response = requests.post(url)

        if response.status_code != 200:
            print(f"Failed to create Qdrant snapshot: {response.text}")
            return None

        snapshot_info = response.json()
        snapshot_name = snapshot_info.get("result", {}).get("name")

        if not snapshot_name:
            print("Failed to get snapshot name")
            return None

        # Download snapshot
        download_url = f"{url}/{snapshot_name}"
        snapshot_response = requests.get(download_url)

        if snapshot_response.status_code != 200:
            print(f"Failed to download snapshot: {snapshot_response.text}")
            return None

        # Save to backups folder
        BACKUP_DIR.mkdir(exist_ok=True)
        snapshot_path = BACKUP_DIR / f"{backup_name}_qdrant.snapshot"

        with open(snapshot_path, 'wb') as f:
            f.write(snapshot_response.content)

        print(f"Qdrant snapshot saved to: {snapshot_path}")

        # Clean up snapshot on server
        requests.delete(download_url)

        return str(snapshot_path)

    except Exception as e:
        print(f"Qdrant backup error: {e}")
        return None


def restore_qdrant(snapshot_path: str):
    """Restore Qdrant collection from snapshot."""
    if not Path(snapshot_path).exists():
        print(f"Snapshot not found: {snapshot_path}")
        return False

    client = get_qdrant_client()
    if not client:
        print("Qdrant not available, cannot restore.")
        return False

    try:
        # Delete existing collection if exists
        try:
            client.delete_collection(QDRANT_COLLECTION)
            print(f"Deleted existing collection: {QDRANT_COLLECTION}")
        except:
            pass

        # Upload snapshot via REST API
        url = f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections/{QDRANT_COLLECTION}/snapshots/upload"

        with open(snapshot_path, 'rb') as f:
            files = {'snapshot': f}
            response = requests.post(url, files=files)

        if response.status_code == 200:
            print(f"Qdrant collection restored from: {snapshot_path}")
            return True
        else:
            print(f"Failed to restore Qdrant: {response.text}")
            return False

    except Exception as e:
        print(f"Qdrant restore error: {e}")
        return False


def get_db_stats():
    """Get database statistics."""
    if not DB_PATH.exists():
        return None

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM ideas")
    total_ideas = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT domain) FROM ideas")
    total_domains = cursor.fetchone()[0]

    cursor.execute("SELECT MAX(weighted_score) FROM ideas")
    best_score = cursor.fetchone()[0] or 0

    cursor.execute("SELECT MIN(created_at), MAX(created_at) FROM ideas")
    date_range = cursor.fetchone()

    conn.close()

    # Add Qdrant stats
    qdrant_stats = get_qdrant_stats()

    return {
        "total_ideas": total_ideas,
        "total_domains": total_domains,
        "best_score": round(best_score, 1),
        "date_range": {
            "first": date_range[0],
            "last": date_range[1]
        },
        "qdrant": qdrant_stats
    }


def export_all_ideas():
    """Export all ideas to JSON."""
    if not DB_PATH.exists():
        return []

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, session_id, domain, mode, concept_name, description,
               target_audience, differentiation, scores, weighted_score, created_at
        FROM ideas ORDER BY weighted_score DESC
    """)

    ideas = []
    for row in cursor.fetchall():
        idea = dict(row)
        # Parse JSON fields
        if idea.get('scores'):
            try:
                idea['scores'] = json.loads(idea['scores'])
            except:
                pass
        if idea.get('differentiation'):
            try:
                idea['differentiation'] = json.loads(idea['differentiation'])
            except:
                pass
        ideas.append(idea)

    conn.close()
    return ideas


def backup_database(backup_name: str = None, include_qdrant: bool = True):
    """Create a backup of the database and optionally Qdrant."""
    if not DB_PATH.exists():
        print("No database found to backup.")
        return None

    # Create backup directory
    BACKUP_DIR.mkdir(exist_ok=True)

    # Generate backup name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if backup_name:
        backup_filename = f"{backup_name}_{timestamp}"
    else:
        backup_filename = f"ideation_backup_{timestamp}"

    # Copy database file
    db_backup_path = BACKUP_DIR / f"{backup_filename}.db"
    shutil.copy2(DB_PATH, db_backup_path)
    print(f"SQLite backed up to: {db_backup_path}")

    # Backup Qdrant
    qdrant_path = None
    if include_qdrant:
        qdrant_path = backup_qdrant(backup_filename)

    # Export to JSON
    ideas = export_all_ideas()
    stats = get_db_stats()

    json_backup = {
        "backup_info": {
            "created_at": datetime.now().isoformat(),
            "source": str(DB_PATH),
            "qdrant_snapshot": qdrant_path,
            "stats": stats
        },
        "ideas": ideas
    }

    json_backup_path = BACKUP_DIR / f"{backup_filename}.json"
    with open(json_backup_path, 'w') as f:
        json.dump(json_backup, f, indent=2, default=str)
    print(f"JSON exported to: {json_backup_path}")

    return {
        "db_path": str(db_backup_path),
        "json_path": str(json_backup_path),
        "qdrant_path": qdrant_path,
        "stats": stats
    }


def list_backups():
    """List all existing backups."""
    if not BACKUP_DIR.exists():
        print("No backups found.")
        return []

    backups = []
    for f in sorted(BACKUP_DIR.glob("*.db"), reverse=True):
        size = f.stat().st_size / 1024  # KB
        backups.append({
            "name": f.name,
            "size_kb": round(size, 1),
            "created": datetime.fromtimestamp(f.stat().st_mtime).isoformat()
        })

    return backups


def restore_backup(backup_name: str, include_qdrant: bool = True):
    """Restore database and optionally Qdrant from backup."""
    backup_path = BACKUP_DIR / backup_name

    if not backup_path.exists():
        print(f"Backup not found: {backup_path}")
        return False

    # Create data directory if needed
    DATA_DIR.mkdir(exist_ok=True)

    # Backup current database first
    if DB_PATH.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pre_restore_backup = BACKUP_DIR / f"pre_restore_{timestamp}.db"
        shutil.copy2(DB_PATH, pre_restore_backup)
        print(f"Current database backed up to: {pre_restore_backup}")

    # Restore SQLite
    shutil.copy2(backup_path, DB_PATH)
    print(f"SQLite restored from: {backup_path}")

    # Restore Qdrant if snapshot exists
    if include_qdrant:
        # Try to find matching Qdrant snapshot
        base_name = backup_name.replace('.db', '')
        qdrant_snapshot = BACKUP_DIR / f"{base_name}_qdrant.snapshot"

        if qdrant_snapshot.exists():
            restore_qdrant(str(qdrant_snapshot))
        else:
            print(f"No Qdrant snapshot found at: {qdrant_snapshot}")

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Universal Ideation v3.2 Backup Tool")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Backup command
    backup_parser = subparsers.add_parser("backup", help="Create a backup")
    backup_parser.add_argument("-n", "--name", help="Backup name prefix")
    backup_parser.add_argument("--no-qdrant", action="store_true", help="Skip Qdrant backup")

    # List command
    list_parser = subparsers.add_parser("list", help="List all backups")

    # Restore command
    restore_parser = subparsers.add_parser("restore", help="Restore from backup")
    restore_parser.add_argument("backup_file", help="Backup filename to restore")
    restore_parser.add_argument("--no-qdrant", action="store_true", help="Skip Qdrant restore")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show database statistics")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export all ideas to JSON")
    export_parser.add_argument("-o", "--output", help="Output file path")

    args = parser.parse_args()

    if args.command == "backup":
        include_qdrant = not getattr(args, 'no_qdrant', False)
        result = backup_database(args.name, include_qdrant=include_qdrant)
        if result:
            print(f"\nBackup complete!")
            print(f"  SQLite: {result['db_path']}")
            if result.get('qdrant_path'):
                print(f"  Qdrant: {result['qdrant_path']}")
            print(f"  JSON: {result['json_path']}")
            print(f"\nStats:")
            print(f"  Ideas: {result['stats']['total_ideas']}")
            print(f"  Domains: {result['stats']['total_domains']}")
            print(f"  Best score: {result['stats']['best_score']}")
            if result['stats'].get('qdrant'):
                qdrant = result['stats']['qdrant']
                if not qdrant.get('error'):
                    print(f"  Qdrant vectors: {qdrant.get('vectors_count', 'N/A')}")

    elif args.command == "list":
        backups = list_backups()
        if backups:
            print(f"\nFound {len(backups)} SQLite backups:\n")
            for b in backups:
                print(f"  {b['name']} ({b['size_kb']} KB) - {b['created']}")
            # Check for Qdrant snapshots
            qdrant_snapshots = list(BACKUP_DIR.glob("*.snapshot")) if BACKUP_DIR.exists() else []
            if qdrant_snapshots:
                print(f"\nQdrant snapshots ({len(qdrant_snapshots)}):")
                for s in qdrant_snapshots:
                    size = s.stat().st_size / 1024
                    print(f"  {s.name} ({round(size, 1)} KB)")
        else:
            print("No backups found.")

    elif args.command == "restore":
        include_qdrant = not getattr(args, 'no_qdrant', False)
        restore_backup(args.backup_file, include_qdrant=include_qdrant)

    elif args.command == "stats":
        stats = get_db_stats()
        if stats:
            print(f"\nDatabase Statistics:")
            print(f"  Total ideas: {stats['total_ideas']}")
            print(f"  Total domains: {stats['total_domains']}")
            print(f"  Best score: {stats['best_score']}")
            print(f"  Date range: {stats['date_range']['first']} to {stats['date_range']['last']}")
            if stats.get('qdrant'):
                print(f"\nQdrant Statistics:")
                qdrant = stats['qdrant']
                if qdrant.get('error'):
                    print(f"  Status: Not connected")
                else:
                    print(f"  Vectors: {qdrant.get('vectors_count', 'N/A')}")
                    print(f"  Status: {qdrant.get('status', 'N/A')}")
        else:
            print("No database found.")

    elif args.command == "export":
        ideas = export_all_ideas()
        output_path = args.output or f"ideas_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, 'w') as f:
            json.dump(ideas, f, indent=2, default=str)
        print(f"Exported {len(ideas)} ideas to: {output_path}")

    else:
        parser.print_help()
