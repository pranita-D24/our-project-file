import os
import json
import logging
import hashlib
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Any, Optional

import yaml
import fitz  # PyMuPDF

# ═══════════════════════════════════════════════════════════
# CONFIGURATION & EXCEPTIONS
# ═══════════════════════════════════════════════════════════

class InvalidPDFError(Exception):
    """Raised when a PDF file is corrupt or unreadable."""
    pass

class Stage1Config:
    def __init__(self, config_path: str = "pipeline_config.yaml"):
        p = Path(config_path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(p, "r") as f:
            data = yaml.safe_load(f)
            
        self.drawings_root = Path(data["drawings_root"])
        self.output_root   = Path(data["output_root"])
        self.v1_suffix     = data["v1_suffix"]
        self.v2_suffix     = data["v2_suffix"]
        self.max_workers   = data.get("max_workers", 8)
        self.batch_size    = data.get("batch_size", 500)
        
        # Ensure output directory exists
        self.output_root.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════

def setup_logging(drawing_id: str = "INIT"):
    """Configures logging for Stage 1."""
    logging.basicConfig(
        level=logging.INFO,
        format=f"[STAGE1] [%(drawing_id)s] %(message)s"
    )

def get_drawing_id(filepath: str) -> str:
    """Extracts drawing ID from filename."""
    return Path(filepath).stem.replace("_V1", "").replace("_V2", "")

# Custom filter to add drawing_id to log records
class DrawingIDFilter(logging.Filter):
    def __init__(self, drawing_id: str):
        super().__init__()
        self.drawing_id = drawing_id
    def filter(self, record):
        record.drawing_id = self.drawing_id
        return True

logger = logging.getLogger("Stage1")
logger.setLevel(logging.INFO)
# Prevent double logging if re-imported
if not logger.handlers:
    sh = logging.StreamHandler()
    formatter = logging.Formatter("[STAGE1] [%(drawing_id)s] %(message)s")
    sh.setFormatter(formatter)
    logger.addHandler(sh)

def log_with_id(drawing_id: str, message: str, level: int = logging.INFO):
    """Helper to log with drawing_id context."""
    logger.log(level, message, extra={"drawing_id": drawing_id})

# ═══════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ═══════════════════════════════════════════════════════════

def compute_sha256(filepath: str) -> str:
    """
    Computes SHA-256 hash of a file in 64KB chunks.
    Args:
        filepath: Path to the file.
    Returns:
        Hex digest of the file.
    """
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Read in 64KB chunks
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def identity_check(path_v1: str, path_v2: str) -> Dict[str, Any]:
    """
    Compares two files via SHA-256 hash.
    Args:
        path_v1: Path to V1 PDF.
        path_v2: Path to V2 PDF.
    Returns:
        Identity result dictionary.
    """
    h1 = compute_sha256(path_v1)
    h2 = compute_sha256(path_v2)
    
    drawing_id = get_drawing_id(path_v1)
    
    if h1 == h2:
        log_with_id(drawing_id, "Hashes match — skipping processing")
        return {
            "status": "IDENTICAL",
            "score": 100,
            "ssim": 1.0,
            "changed_pixels": 0.0,
            "skip_processing": True,
            "hash_v1": h1,
            "hash_v2": h2
        }
    else:
        return {
            "status": "PROCEED",
            "score": None,
            "ssim": None,
            "changed_pixels": None,
            "skip_processing": False,
            "hash_v1": h1,
            "hash_v2": h2
        }

def open_pdf(filepath: str) -> fitz.Document:
    """
    Opens a PDF using PyMuPDF and validates it.
    Args:
        filepath: Path to PDF.
    Returns:
        PyMuPDF document object.
    """
    drawing_id = get_drawing_id(filepath)
    p = Path(filepath)
    if not p.exists():
        log_with_id(drawing_id, f"File not found: {filepath}", logging.ERROR)
        raise InvalidPDFError(f"File not found: {filepath}")
    
    try:
        doc = fitz.open(filepath)
        if doc.is_closed or doc.is_encrypted:
            raise ValueError("Encrypted or closed document")
        log_with_id(drawing_id, f"Opened PDF: {len(doc)} pages")
        return doc
    except Exception as e:
        log_with_id(drawing_id, f"InvalidPDFError: {e}", logging.ERROR)
        raise InvalidPDFError(str(e))

def ingest_pair(path_v1: str, path_v2: str) -> Dict[str, Any]:
    """
    Main entry point for Stage 1 pair ingestion.
    """
    drawing_id = get_drawing_id(path_v1)
    
    try:
        # Step A & B: Identity Check
        res = identity_check(path_v1, path_v2)
        if res["skip_processing"]:
            return res
        
        # Step C: Open PDFs
        doc1 = open_pdf(path_v1)
        doc2 = open_pdf(path_v2)
        
        cnt1 = len(doc1)
        cnt2 = len(doc2)
        mismatch = (cnt1 != cnt2)
        
        if mismatch:
            log_with_id(drawing_id, f"Page count mismatch: V1={cnt1}, V2={cnt2}", logging.WARNING)
        
        return {
            "status": "PROCEED",
            "skip_processing": False,
            "hash_v1": res["hash_v1"],
            "hash_v2": res["hash_v2"],
            "doc_v1": doc1,
            "doc_v2": doc2,
            "page_count_v1": cnt1,
            "page_count_v2": cnt2,
            "page_count_mismatch": mismatch,
            "error_message": None
        }
    except InvalidPDFError as e:
        return {
            "status": "ERROR",
            "skip_processing": False,
            "hash_v1": None,
            "hash_v2": None,
            "error_message": str(e)
        }
    except Exception as e:
        log_with_id(drawing_id, f"Unexpected error: {e}", logging.ERROR)
        return {
            "status": "ERROR",
            "skip_processing": False,
            "hash_v1": None,
            "hash_v2": None,
            "error_message": str(e)
        }

def save_manifest(results: List[Dict[str, Any]], config: Stage1Config):
    """
    Saves the results to an atomic JSON manifest.
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"RUN_{timestamp}"
    
    summary = {
        "run_id": run_id,
        "total_pairs": len(results),
        "identical_count": sum(1 for r in results if r.get("status") == "IDENTICAL"),
        "proceed_count": sum(1 for r in results if r.get("status") == "PROCEED"),
        "error_count": sum(1 for r in results if r.get("status") == "ERROR")
    }
    
    # Process pairs for serialization (remove document objects)
    serializable_pairs = []
    for r in results:
        pair_data = r.copy()
        # Remove non-serializable fitz.Document objects
        if "doc_v1" in pair_data: del pair_data["doc_v1"]
        if "doc_v2" in pair_data: del pair_data["doc_v2"]
        serializable_pairs.append(pair_data)
        
    manifest = {
        **summary,
        "pairs": serializable_pairs
    }
    
    file_name = "stage1_manifest.json"
    target_path = config.output_root / file_name
    
    # Versioning: If exists, add timestamp
    if target_path.exists():
        file_name = f"stage1_manifest_{timestamp}.json"
        target_path = config.output_root / file_name
        
    tmp_path = target_path.with_suffix(".tmp")
    
    try:
        with open(tmp_path, "w") as f:
            json.dump(manifest, f, indent=2)
        
        # Atomic rename
        os.replace(tmp_path, target_path)
        print(f"\nManifest saved to: {target_path}")
    except Exception as e:
        print(f"Failed to save manifest: {e}")

def run_batch(pairs: List[Tuple[str, str]], config: Stage1Config) -> List[Dict[str, Any]]:
    """
    Processes multiple pairs in parallel.
    """
    results = []
    
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        future_to_pair = {executor.submit(ingest_pair, p[0], p[1]): p for p in pairs}
        for future in as_completed(future_to_pair):
            pair = future_to_pair[future]
            drawing_id = get_drawing_id(pair[0])
            try:
                data = future.result()
                # Add metadata for manifest
                data["drawing_id"] = drawing_id
                data["path_v1"] = pair[0]
                data["path_v2"] = pair[1]
                results.append(data)
            except Exception as e:
                results.append({
                    "drawing_id": drawing_id,
                    "path_v1": pair[0],
                    "path_v2": pair[1],
                    "status": "ERROR",
                    "error_message": str(e)
                })

    # Print Summary
    identical = sum(1 for r in results if r["status"] == "IDENTICAL")
    proceed   = sum(1 for r in results if r["status"] == "PROCEED")
    errors    = sum(1 for r in results if r["status"] == "ERROR")
    
    print("\n" + "="*40)
    print(f"Total Pairs   : {len(results)}")
    print(f"Identical     : {identical}  (skipped)")
    print(f"To Process    : {proceed}")
    print(f"Errors        : {errors}")
    print("="*40 + "\n")
    
    save_manifest(results, config)
    return results

if __name__ == "__main__":
    # Example usage if run directly
    cfg = Stage1Config()
    # Logic to find pairs in drawings_root could go here
    pass
