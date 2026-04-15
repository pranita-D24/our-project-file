# verify_stage3_4.py
import sys, os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import pandas as pd
from database_spatial import SpatialDatabaseManager
from parquet_io import ParquetStore
from comparator import _match_components

def test_spatial_db():
    print("--- Testing Spatial DB ---")
    if os.path.exists("test_spatial.db"):
        os.remove("test_spatial.db")
    db = SpatialDatabaseManager("sqlite:///test_spatial.db")
    
    # 1. Computed Centroid Test
    elements = [
        {"id": "E1", "drawing_id": "DWG1", "type": "component", 
         "bbox_x": 100, "bbox_y": 100, "bbox_w": 20, "bbox_h": 20, "value": 10.0},
    ]
    db.bulk_save_elements(elements)
    
    # Verify centroid computation (110, 110)
    res = db.find_within_radius(110, 110, 5)
    print(f"Centroid Match: {len(res)} (Expected: 1)")
    if res:
        print(f"Match ID: {res[0]['id']} at {res[0]['centroid']}")
    
    # 2. Rtree Removal Test
    db.local_rtree.remove("E1")
    res = db.find_within_radius(110, 110, 5)
    print(f"Removal Match: {len(res)} (Expected: 0)")

def test_parquet():
    print("\n--- Testing Parquet ---")
    store = ParquetStore("temp/test_parquet")
    
    elements = [
        {"id": "E1", "drawing_id": "ABCD-001", "type": "component", 
         "bbox_x": 10, "bbox_y": 10, "bbox_w": 5, "bbox_h": 5, "value": 1.0, 
         "centroid_x": 12.5, "centroid_y": 12.5, "confidence": 0.9, 
         "metadata_json": "{}", "processed_at": pd.Timestamp.now().floor('ms'),
         "text": ""},
    ]
    
    store.save_elements(elements)
    
    # Verify Partitioning: should be in ROOT/prefix=ABCD/revision=V1
    path = "temp/test_parquet/prefix=ABCD/revision=V1"
    print(f"Partition Path Exists: {os.path.exists(path)}")
    
    # Verify Filter Pushdown Read
    df = store.load_elements(drawing_id="ABCD-001")
    print(f"Loaded Rows: {len(df)} (Expected: 1)")
    if not df.empty:
        # Check if column exists, as partition columns might be handled differently by read_table
        p_col = 'prefix' if 'prefix' in df.columns else 'ANY'
        print(f"Loaded Prefix Status: {p_col}")

def test_matcher_hierarchy():
    print("\n--- Testing Matcher Hierarchy ---")
    # Mocks for matching: need hu, sig, cnt, shape
    m_hu = [0.0] * 7
    m_sig = [0.0] * 32
    m_cnt = np.zeros((10, 1, 2), dtype=np.int32)
    
    c1 = [{"id": "1", "centroid": (100, 100), "area": 100, "bbox": (95, 95, 10, 10), 
           "text": "10", "hu": m_hu, "sig": m_sig, "cnt": m_cnt, "shape": "complex"}]
    c2 = [{"id": "2", "centroid": (100, 100), "area": 100, "bbox": (95, 95, 10, 10), 
           "text": "10", "hu": m_hu, "sig": m_sig, "cnt": m_cnt, "shape": "complex"}]
    
    # 1. MATCH case
    g = np.zeros((200, 200), dtype=np.uint8)
    matched, _, _ = _match_components(c1, c2, g, g, move_threshold_px=15.0)
    if matched:
        print(f"Hierarchy Match: {matched[0][0]} (Expected: MATCH)")

    # 2. Relaxed Gating (1.5x)
    c2_moved = [{"id": "2", "centroid": (120, 100), "area": 100, "bbox": (115, 95, 10, 10), 
                 "text": "10", "hu": m_hu, "sig": m_sig, "cnt": m_cnt, "shape": "complex"}]
    matched, _, _ = _match_components(c1, c2_moved, g, g, move_threshold_px=15.0)
    if matched:
        print(f"Gating Match: {matched[0][0]} (Expected: MOVED)")

if __name__ == "__main__":
    test_spatial_db()
    test_parquet()
    test_matcher_hierarchy()
