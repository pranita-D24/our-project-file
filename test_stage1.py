import os
import shutil
import json
import logging
from pathlib import Path
import fitz
import pytest
from stage1_ingest import (
    compute_sha256, identity_check, open_pdf, ingest_pair, 
    run_batch, Stage1Config, InvalidPDFError
)

# ═══════════════════════════════════════════════════════════
# TEST SETUP
# ═══════════════════════════════════════════════════════════

TEST_ROOT = Path("c:/Trivim Internship/engineering_comparison_system/test_data")
TEST_OUT = TEST_ROOT / "output"

def create_test_pdf(path: Path, content: str):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), content)
    doc.save(path)
    doc.close()

def setup_test_env_plain():
    if TEST_ROOT.exists():
        shutil.rmtree(TEST_ROOT)
    TEST_ROOT.mkdir(parents=True, exist_ok=True)
    TEST_OUT.mkdir(parents=True, exist_ok=True)
    
    # Create valid PDFs
    create_test_pdf(TEST_ROOT / "A_V1.pdf", "Content A")
    shutil.copy(TEST_ROOT / "A_V1.pdf", TEST_ROOT / "A_V2.pdf") # Guaranteed SHA-256 match
    create_test_pdf(TEST_ROOT / "B_V2.pdf", "Content B") # Different content
    
    # Create corrupt PDF
    with open(TEST_ROOT / "corrupt_V1.pdf", "w") as f:
        f.write("Not a PDF")
        
    # Create a mock config
    import yaml
    config_data = {
        "drawings_root": str(TEST_ROOT),
        "output_root": str(TEST_OUT),
        "v1_suffix": "_V1.pdf",
        "v2_suffix": "_V2.pdf",
        "max_workers": 4,
        "batch_size": 10
    }
    with open(TEST_ROOT / "test_config.yaml", "w") as f:
        yaml.dump(config_data, f)

# ═══════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════

def test_identical_files():
    """Test 1 — Identical files: hashes match, skip processing."""
    path_v1 = TEST_ROOT / "A_V1.pdf"
    path_v2 = TEST_ROOT / "A_V2.pdf"
    
    result = ingest_pair(str(path_v1), str(path_v2))
    
    assert result["status"] == "IDENTICAL"
    assert result["score"] == 100
    assert result["skip_processing"] is True
    assert result["hash_v1"] == result["hash_v2"]

def test_different_files():
    """Test 2 — Different files: proceed to open."""
    path_v1 = TEST_ROOT / "A_V1.pdf"
    path_v2 = TEST_ROOT / "B_V2.pdf"
    
    result = ingest_pair(str(path_v1), str(path_v2))
    
    assert result["status"] == "PROCEED"
    assert result["skip_processing"] is False
    assert result["doc_v1"] is not None
    assert result["hash_v1"] != result["hash_v2"]
    assert result["page_count_v1"] == 1

def test_corrupt_file():
    """Test 3 — Corrupt file: InvalidPDFError raised and captured."""
    path_v1 = TEST_ROOT / "corrupt_V1.pdf"
    path_v2 = TEST_ROOT / "A_V2.pdf"
    
    # ingest_pair catches it and returns ERROR status
    result = ingest_pair(str(path_v1), str(path_v2))
    
    assert result["status"] == "ERROR"
    assert any(x in str(result["error_message"]).lower() for x in ["invalid", "format error", "failed to open"])

def test_batch_runner():
    """Test 4 — Batch runner: multiple pairs, summary, manifest."""
    config = Stage1Config(str(TEST_ROOT / "test_config.yaml"))
    
    pairs = [
        (str(TEST_ROOT / "A_V1.pdf"), str(TEST_ROOT / "A_V2.pdf")),
        (str(TEST_ROOT / "A_V1.pdf"), str(TEST_ROOT / "B_V2.pdf")),
        (str(TEST_ROOT / "corrupt_V1.pdf"), str(TEST_ROOT / "A_V2.pdf")),
        (str(TEST_ROOT / "A_V1.pdf"), str(TEST_ROOT / "A_V1.pdf")), # Another identical
        (str(TEST_ROOT / "A_V1.pdf"), str(TEST_ROOT / "A_V2.pdf")), # Another identical
    ]
    
    results = run_batch(pairs, config)
    
    assert isinstance(results, list)
    assert len(results) == 5
    
    # Check if manifest was saved
    manifest_path = TEST_OUT / "stage1_manifest.json"
    assert manifest_path.exists()
    
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
        assert manifest["total_pairs"] == 5
        assert "run_id" in manifest
        assert len(manifest["pairs"]) == 5
        # Verify drawing_id exists
        assert manifest["pairs"][0]["drawing_id"] in ["A", "corrupt"]

def run_all_tests():
    setup_test_env_plain()
    print("Running Tests...\n")
    try:
        test_identical_files()
        print("Test 1 Passed: Identical files")
        test_different_files()
        print("Test 2 Passed: Different files")
        test_corrupt_file()
        print("Test 3 Passed: Corrupt file")
        test_batch_runner()
        print("Test 4 Passed: Batch runner")
        print("\nALL STAGE 1 TESTS PASSED SUCCESSFULLY!")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # cleanup_test_env_plain()
        pass

if __name__ == "__main__":
    run_all_tests()
