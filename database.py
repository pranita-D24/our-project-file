# database.py
# Industry Grade SQLite Database Manager
# Handles all data storage for the system

import sqlite3
import os
import json
import uuid
import logging
from datetime import datetime
from config import DATABASE_DIR

# ══════════════════════════════════════
# LOGGING SETUP
# ══════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(
            os.path.join(DATABASE_DIR, "system.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════
# DATABASE PATH
# ══════════════════════════════════════
DB_PATH = os.path.join(DATABASE_DIR, "drawings.db")


# ══════════════════════════════════════
# DATABASE MANAGER CLASS
# ══════════════════════════════════════
class DatabaseManager:

    def __init__(self):
        self.db_path = DB_PATH
        self._initialize_database()
        logger.info("Database initialized successfully")

    def _get_connection(self):
        """Get database connection with row factory"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _initialize_database(self):
        """Create all tables if they don't exist"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # ── Table 1: Drawings ──
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS drawings (
                    id          TEXT PRIMARY KEY,
                    name        TEXT NOT NULL,
                    description TEXT,
                    drawing_no  TEXT UNIQUE,
                    category    TEXT,
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL,
                    status      TEXT DEFAULT 'active',
                    metadata    TEXT
                )
            """)

            # ── Table 2: Versions ──
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS versions (
                    id           TEXT PRIMARY KEY,
                    drawing_id   TEXT NOT NULL,
                    version_no   TEXT NOT NULL,
                    pdf_path     TEXT NOT NULL,
                    image_paths  TEXT,
                    page_count   INTEGER DEFAULT 1,
                    uploaded_by  TEXT,
                    uploaded_at  TEXT NOT NULL,
                    notes        TEXT,
                    status       TEXT DEFAULT 'active',
                    FOREIGN KEY (drawing_id)
                        REFERENCES drawings(id),
                    UNIQUE(drawing_id, version_no)
                )
            """)

            # ── Table 3: Comparisons ──
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS comparisons (
                    id              TEXT PRIMARY KEY,
                    drawing_id      TEXT NOT NULL,
                    version_1_id    TEXT NOT NULL,
                    version_2_id    TEXT NOT NULL,
                    similarity      REAL,
                    added_count     INTEGER DEFAULT 0,
                    removed_count   INTEGER DEFAULT 0,
                    modified_count  INTEGER DEFAULT 0,
                    moved_count     INTEGER DEFAULT 0,
                    verdict         TEXT,
                    report_path     TEXT,
                    json_path       TEXT,
                    compared_at     TEXT NOT NULL,
                    processing_time REAL,
                    status          TEXT DEFAULT 'completed',
                    FOREIGN KEY (drawing_id)
                        REFERENCES drawings(id),
                    FOREIGN KEY (version_1_id)
                        REFERENCES versions(id),
                    FOREIGN KEY (version_2_id)
                        REFERENCES versions(id)
                )
            """)

            # ── Table 4: Objects ──
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS objects (
                    id            TEXT PRIMARY KEY,
                    comparison_id TEXT NOT NULL,
                    object_id     INTEGER,
                    version       TEXT,
                    change_type   TEXT,
                    area          REAL,
                    centroid_x    REAL,
                    centroid_y    REAL,
                    bbox_x        INTEGER,
                    bbox_y        INTEGER,
                    bbox_w        INTEGER,
                    bbox_h        INTEGER,
                    shape_type    TEXT,
                    ocr_text      TEXT,
                    FOREIGN KEY (comparison_id)
                        REFERENCES comparisons(id)
                )
            """)

            # ── Table 5: Audit Log ──
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id         TEXT PRIMARY KEY,
                    action     TEXT NOT NULL,
                    entity     TEXT,
                    entity_id  TEXT,
                    details    TEXT,
                    timestamp  TEXT NOT NULL,
                    status     TEXT DEFAULT 'success'
                )
            """)

            # ── Indexes for faster queries ──
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS
                idx_versions_drawing
                ON versions(drawing_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS
                idx_comparisons_drawing
                ON comparisons(drawing_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS
                idx_objects_comparison
                ON objects(comparison_id)
            """)

            conn.commit()

        except Exception as e:
            logger.error(f"Database init error: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()

    # ══════════════════════════════════
    # DRAWING OPERATIONS
    # ══════════════════════════════════
    def add_drawing(self, name, drawing_no=None,
                    description=None, category=None,
                    metadata=None):
        """Add a new drawing, or return existing ID if drawing_no already exists"""
        conn = self._get_connection()
        try:
            if drawing_no:
                existing = conn.execute("""
                    SELECT id FROM drawings WHERE drawing_no = ?
                """, (drawing_no,)).fetchone()
                if existing:
                    logger.info(
                        f"Drawing {drawing_no} already exists, "
                        f"returning existing ID")
                    return existing["id"]

            drawing_id = str(uuid.uuid4())[:8].upper()
            now        = datetime.now().isoformat()

            conn.execute("""
                INSERT INTO drawings
                (id, name, description, drawing_no,
                 category, created_at, updated_at, metadata)
                VALUES (?,?,?,?,?,?,?,?)
            """, (drawing_id, name, description,
                  drawing_no, category, now, now,
                  json.dumps(metadata or {})))

            conn.commit()
            self._log_action(conn, "ADD_DRAWING",
                             "drawing", drawing_id,
                             f"Added: {name}")
            conn.commit()
            logger.info(
                f"Drawing added: {name} [{drawing_id}]")
            return drawing_id

        except Exception as e:
            logger.error(f"Error adding drawing: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_drawing(self, drawing_id):
        """Get drawing by ID"""
        conn = self._get_connection()
        try:
            row = conn.execute("""
                SELECT * FROM drawings WHERE id = ?
            """, (drawing_id,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_all_drawings(self, status="active"):
        """Get all drawings"""
        conn = self._get_connection()
        try:
            rows = conn.execute("""
                SELECT * FROM drawings
                WHERE status = ?
                ORDER BY created_at DESC
            """, (status,)).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def search_drawings(self, query):
        """Search drawings by name or number"""
        conn = self._get_connection()
        try:
            rows = conn.execute("""
                SELECT * FROM drawings
                WHERE name LIKE ?
                OR drawing_no LIKE ?
                OR description LIKE ?
                ORDER BY created_at DESC
            """, (f"%{query}%",
                  f"%{query}%",
                  f"%{query}%")).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def update_drawing(self, drawing_id, **kwargs):
        """Update drawing fields"""
        conn = self._get_connection()
        try:
            kwargs["updated_at"] = datetime.now().isoformat()
            fields = ", ".join(
                [f"{k} = ?" for k in kwargs.keys()])
            values = list(kwargs.values()) + [drawing_id]
            conn.execute(f"""
                UPDATE drawings SET {fields}
                WHERE id = ?
            """, values)
            conn.commit()
            logger.info(f"Drawing updated: {drawing_id}")
        except Exception as e:
            logger.error(f"Error updating drawing: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()

    def delete_drawing(self, drawing_id):
        """Soft delete drawing"""
        self.update_drawing(drawing_id, status="deleted")
        logger.info(f"Drawing deleted: {drawing_id}")

    # ══════════════════════════════════
    # VERSION OPERATIONS
    # ══════════════════════════════════
    def add_version(self, drawing_id, version_no,
                    pdf_path, image_paths=None,
                    page_count=1, uploaded_by=None,
                    notes=None):
        """Add a new version, or return existing ID if already exists"""
        conn = self._get_connection()
        try:
            existing = conn.execute("""
                SELECT id FROM versions
                WHERE drawing_id = ? AND version_no = ?
            """, (drawing_id, version_no)).fetchone()
            if existing:
                logger.info(
                    f"Version {version_no} already exists "
                    f"for drawing {drawing_id}, returning existing ID")
                return existing["id"]

            version_id = str(uuid.uuid4())[:8].upper()
            now        = datetime.now().isoformat()

            conn.execute("""
                INSERT INTO versions
                (id, drawing_id, version_no, pdf_path,
                 image_paths, page_count, uploaded_by,
                 uploaded_at, notes)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, (version_id, drawing_id, version_no,
                  pdf_path,
                  json.dumps(image_paths or []),
                  page_count, uploaded_by, now, notes))

            conn.commit()
            self._log_action(conn, "ADD_VERSION",
                             "version", version_id,
                             f"Drawing:{drawing_id} Ver:{version_no}")
            conn.commit()
            logger.info(
                f"Version added: {version_no} for {drawing_id}")
            return version_id

        except Exception as e:
            logger.error(f"Error adding version: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_versions(self, drawing_id):
        """Get all versions of a drawing"""
        conn = self._get_connection()
        try:
            rows = conn.execute("""
                SELECT * FROM versions
                WHERE drawing_id = ?
                AND status = 'active'
                ORDER BY uploaded_at ASC
            """, (drawing_id,)).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_version(self, version_id):
        """Get specific version"""
        conn = self._get_connection()
        try:
            row = conn.execute("""
                SELECT * FROM versions WHERE id = ?
            """, (version_id,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_latest_version(self, drawing_id):
        """Get latest version of a drawing"""
        conn = self._get_connection()
        try:
            row = conn.execute("""
                SELECT * FROM versions
                WHERE drawing_id = ?
                AND status = 'active'
                ORDER BY uploaded_at DESC
                LIMIT 1
            """, (drawing_id,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    # ══════════════════════════════════
    # COMPARISON OPERATIONS
    # ══════════════════════════════════
    def add_comparison(self, drawing_id, version_1_id,
                       version_2_id, similarity=None,
                       added=0, removed=0, modified=0,
                       moved=0, verdict=None,
                       report_path=None, json_path=None,
                       processing_time=None):
        """Save comparison results"""
        conn = self._get_connection()
        try:
            comparison_id = str(uuid.uuid4())[:8].upper()
            now           = datetime.now().isoformat()

            conn.execute("""
                INSERT INTO comparisons
                (id, drawing_id, version_1_id, version_2_id,
                 similarity, added_count, removed_count,
                 modified_count, moved_count, verdict,
                 report_path, json_path, compared_at,
                 processing_time)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (comparison_id, drawing_id,
                  version_1_id, version_2_id,
                  similarity, added, removed,
                  modified, moved, verdict,
                  report_path, json_path,
                  now, processing_time))

            conn.commit()
            logger.info(
                f"Comparison saved: {comparison_id}")
            return comparison_id

        except Exception as e:
            logger.error(f"Error saving comparison: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_comparisons(self, drawing_id):
        """Get all comparisons for a drawing"""
        conn = self._get_connection()
        try:
            rows = conn.execute("""
                SELECT c.*,
                    v1.version_no as v1_no,
                    v2.version_no as v2_no
                FROM comparisons c
                JOIN versions v1 ON c.version_1_id = v1.id
                JOIN versions v2 ON c.version_2_id = v2.id
                WHERE c.drawing_id = ?
                ORDER BY c.compared_at DESC
            """, (drawing_id,)).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_comparison(self, comparison_id):
        """Get specific comparison"""
        conn = self._get_connection()
        try:
            row = conn.execute("""
                SELECT * FROM comparisons WHERE id = ?
            """, (comparison_id,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    # ══════════════════════════════════
    # OBJECT OPERATIONS
    # ══════════════════════════════════
    def add_objects(self, comparison_id, objects):
        """Batch insert detected objects"""
        conn = self._get_connection()
        try:
            for obj in objects:
                obj_id = str(uuid.uuid4())[:8].upper()
                conn.execute("""
                    INSERT INTO objects
                    (id, comparison_id, object_id,
                     version, change_type, area,
                     centroid_x, centroid_y,
                     bbox_x, bbox_y, bbox_w, bbox_h,
                     shape_type, ocr_text)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (obj_id, comparison_id,
                      obj.get("object_id"),
                      obj.get("version"),
                      obj.get("change_type"),
                      obj.get("area"),
                      obj.get("centroid_x"),
                      obj.get("centroid_y"),
                      obj.get("bbox_x"),
                      obj.get("bbox_y"),
                      obj.get("bbox_w"),
                      obj.get("bbox_h"),
                      obj.get("shape_type"),
                      obj.get("ocr_text")))
            conn.commit()
            logger.info(
                f"Saved {len(objects)} objects "
                f"for comparison {comparison_id}")
        except Exception as e:
            logger.error(f"Error saving objects: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_objects(self, comparison_id,
                    change_type=None):
        """Get objects for a comparison"""
        conn = self._get_connection()
        try:
            if change_type:
                rows = conn.execute("""
                    SELECT * FROM objects
                    WHERE comparison_id = ?
                    AND change_type = ?
                    ORDER BY object_id
                """, (comparison_id, change_type)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT * FROM objects
                    WHERE comparison_id = ?
                    ORDER BY object_id
                """, (comparison_id,)).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # ══════════════════════════════════
    # STATISTICS
    # ══════════════════════════════════
    def get_statistics(self):
        """Get system wide statistics"""
        conn = self._get_connection()
        try:
            stats = {}

            stats["total_drawings"] = conn.execute("""
                SELECT COUNT(*) FROM drawings
                WHERE status = 'active'
            """).fetchone()[0]

            stats["total_versions"] = conn.execute("""
                SELECT COUNT(*) FROM versions
                WHERE status = 'active'
            """).fetchone()[0]

            stats["total_comparisons"] = conn.execute("""
                SELECT COUNT(*) FROM comparisons
            """).fetchone()[0]

            stats["avg_similarity"] = conn.execute("""
                SELECT ROUND(AVG(similarity), 2)
                FROM comparisons
            """).fetchone()[0]

            row = conn.execute("""
                SELECT
                    SUM(added_count)    as total_added,
                    SUM(removed_count)  as total_removed,
                    SUM(modified_count) as total_modified
                FROM comparisons
            """).fetchone()

            stats["total_changes"] = {
                "total_added"   : row["total_added"]    or 0,
                "total_removed" : row["total_removed"]  or 0,
                "total_modified": row["total_modified"] or 0
            }

            return stats
        finally:
            conn.close()
    # ══════════════════════════════════
    # AUDIT LOG
    # ══════════════════════════════════
    def _log_action(self, conn, action, entity=None,
                    entity_id=None, details=None,
                    status="success"):
        """Internal audit logging"""
        try:
            log_id = str(uuid.uuid4())[:8].upper()
            conn.execute("""
                INSERT INTO audit_log
                (id, action, entity, entity_id,
                 details, timestamp, status)
                VALUES (?,?,?,?,?,?,?)
            """, (log_id, action, entity,
                  entity_id, details,
                  datetime.now().isoformat(), status))
        except Exception as e:
            logger.warning(f"Audit log error: {e}")

    def get_audit_log(self, limit=100):
        """Get recent audit log entries"""
        conn = self._get_connection()
        try:
            rows = conn.execute("""
                SELECT * FROM audit_log
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,)).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


# ══════════════════════════════════════
# TEST DATABASE
# ══════════════════════════════════════
if __name__ == "__main__":

    print("Testing Database...")
    print("=" * 50)

    db = DatabaseManager()

    # Test adding drawing
    drawing_id = db.add_drawing(
        name        = "Gear Assembly Drawing",
        drawing_no  = "PRV73B001",
        description = "Main gear assembly v1",
        category    = "Mechanical"
    )
    print(f"Drawing added : {drawing_id}")

    # Test adding version
    version_id = db.add_version(
        drawing_id = drawing_id,
        version_no = "v1",
        pdf_path   = "uploads/PRV73B001/v1.pdf",
        page_count = 1,
        notes      = "Initial version"
    )
    print(f"Version added : {version_id}")

    # Test getting drawing
    drawing = db.get_drawing(drawing_id)
    print(f"Drawing found : {drawing['name']}")

    # Test statistics
    stats = db.get_statistics()
    print(f"Statistics    : {stats}")

    print()
    print("Database working perfectly ✅")
    print("=" * 50)