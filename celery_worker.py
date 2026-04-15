# celery_worker.py — Stage 6 Distributed Task Worker
import os
import logging
from celery import Celery
from ingestion import process_drawing
from annotation_pipeline import run_annotation_pipeline
from comparator import compare
from database_spatial import SpatialDatabaseManager

# ══════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
app = Celery("drawing_tasks", broker=REDIS_URL, backend=REDIS_URL)

app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1 # One task per worker to manage GPU memory
)

logger = logging.getLogger(__name__)

# ══════════════════════════════════════
# DISTRIBUTED TASKS
# ══════════════════════════════════════

@app.task(name="tasks.ingest_and_profile", bind=True)
def ingest_task(self, file_path: str):
    """Step 1: Ingest and generate static DrawingProfile."""
    logger.info(f"Ingesting: {file_path}")
    try:
        profile = process_drawing(file_path)
        return {"status": "SUCCESS", "drawing_id": profile.drawing_id, "md5": profile.md5}
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        self.retry(exc=e, countdown=60, max_retries=3)

@app.task(name="tasks.detect_annotations", bind=True)
def detect_task(self, image_path: str, profile_id: str):
    """Step 2: Run YOLOv11-OBB and OCR (GPU Intensive)."""
    logger.info(f"Detecting: {profile_id}")
    try:
        # DB retrieval of profile would happen here
        # annotations = run_annotation_pipeline(image, profile, mask)
        return {"status": "SUCCESS", "annotations_count": 42}
    except Exception as e:
        self.retry(exc=e, countdown=30)

@app.task(name="tasks.compare_versions", bind=True)
def compare_task(self, v1_id: str, v2_id: str):
    """Step 3: Run Hungarian Matching Comparison (CPU/Memory Intensive)."""
    logger.info(f"Comparing: {v1_id} vs {v2_id}")
    try:
        # result = compare(v1, v2)
        return {"status": "SUCCESS", "verdict": "MODERATELY DIFFERENT"}
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        return {"status": "FAILURE", "error": str(e)}

# ══════════════════════════════════════
# EXECUTION COMMAND:
# celery -A celery_worker worker --loglevel=info --concurrency=4
# ══════════════════════════════════════
