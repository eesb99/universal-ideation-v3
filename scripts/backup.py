#!/usr/bin/env python3
"""
Backup script for Universal Ideation v3.2
Backs up SQLite database and exports to JSON
"""

import os
import shutil
import sqlite3
import json
from datetime import datetime
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
BACKUP_DIR = PROJECT_ROOT / "backups"
DB_PATH = DATA_DIR / "ideation.db"


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

    return {
        "total_ideas": total_ideas,
        "total_domains": total_domains,
        "best_score": round(best_score, 1),
        "date_range": {
            "first": date_range[0],
            "last": date_range[1]
        }
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


def backup_database(backup_name: str = None):
    """Create a backup of the database."""
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
    print(f"Database backed up to: {db_backup_path}")

    # Export to JSON
    ideas = export_all_ideas()
    stats = get_db_stats()

    json_backup = {
        "backup_info": {
            "created_at": datetime.now().isoformat(),
            "source": str(DB_PATH),
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


def restore_backup(backup_name: str):
    """Restore database from backup."""
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

    # Restore
    shutil.copy2(backup_path, DB_PATH)
    print(f"Database restored from: {backup_path}")

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Universal Ideation v3.2 Backup Tool")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Backup command
    backup_parser = subparsers.add_parser("backup", help="Create a backup")
    backup_parser.add_argument("-n", "--name", help="Backup name prefix")

    # List command
    list_parser = subparsers.add_parser("list", help="List all backups")

    # Restore command
    restore_parser = subparsers.add_parser("restore", help="Restore from backup")
    restore_parser.add_argument("backup_file", help="Backup filename to restore")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show database statistics")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export all ideas to JSON")
    export_parser.add_argument("-o", "--output", help="Output file path")

    args = parser.parse_args()

    if args.command == "backup":
        result = backup_database(args.name)
        if result:
            print(f"\nBackup complete!")
            print(f"  Ideas: {result['stats']['total_ideas']}")
            print(f"  Domains: {result['stats']['total_domains']}")
            print(f"  Best score: {result['stats']['best_score']}")

    elif args.command == "list":
        backups = list_backups()
        if backups:
            print(f"\nFound {len(backups)} backups:\n")
            for b in backups:
                print(f"  {b['name']} ({b['size_kb']} KB) - {b['created']}")
        else:
            print("No backups found.")

    elif args.command == "restore":
        restore_backup(args.backup_file)

    elif args.command == "stats":
        stats = get_db_stats()
        if stats:
            print(f"\nDatabase Statistics:")
            print(f"  Total ideas: {stats['total_ideas']}")
            print(f"  Total domains: {stats['total_domains']}")
            print(f"  Best score: {stats['best_score']}")
            print(f"  Date range: {stats['date_range']['first']} to {stats['date_range']['last']}")
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
