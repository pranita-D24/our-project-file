import os
import glob
from pass1_added import run as run_diff

def run_audit():
    drawings_dir = "Drawings"
    output_base = os.path.join("visuals", "drawings_audit")
    os.makedirs(output_base, exist_ok=True)

    # We want to find pairs. 
    # Example 1: XXX.PDF and XXX - Copy.PDF
    # Example 2: XXX_ori (N).pdf and XXX (N).pdf
    
    files = os.listdir(drawings_dir)
    pdfs = [f for f in files if f.lower().endswith(".pdf")]
    
    # Strategy: Find all files that contain "Copy" or "(2)" or "(1)" etc. 
    # and map them to their likely original.
    
    pairs = []
    
    # 1. Look for PRV series (PRV73B124...)
    # PRV73B124139.PDF vs PRV73B124139 - Copy.PDF
    prv_bases = [f for f in pdfs if " - Copy" not in f and f.startswith("PRV")]
    for base in prv_bases:
        prefix = base.split(".")[0]
        copy_name = f"{prefix} - Copy.PDF"
        if copy_name in pdfs:
            # Usually 'Copy' is the revision, and 'Base' is original. 
            # Consistent with PRV73B124138 where Copy was V1 and Base was V2?
            # User previously said: python pass1_added.py "Drawings\PRV73B124138 - Copy.PDF" "Drawings\PRV73B124138.PDF"
            # So Copy = V1, Original = V2
            pairs.append((os.path.join(drawings_dir, copy_name), os.path.join(drawings_dir, base)))

    # 2. Look for PRQ series
    # PRQ93B101905_ori (1).pdf vs PRQ93B101905 (2).pdf
    # PRQ93B101928_ori (1).pdf vs PRQ93B101928 (1).pdf
    prq_oris = [f for f in pdfs if "_ori" in f]
    for ori in prq_oris:
        # Try to find corresponding non-ori
        base_id = ori.split("_ori")[0]
        # Find files starting with base_id and NOT containing _ori
        candidates = [f for f in pdfs if f.startswith(base_id) and "_ori" not in f]
        if candidates:
            # Pick the one with the highest (N) or just the first one
            pairs.append((os.path.join(drawings_dir, ori), os.path.join(drawings_dir, candidates[0])))

    # 3. Look for drawing 1.pdf, drawing 1-Copy.pdf etc.
    d_bases = [f for f in pdfs if f.startswith("drawing ") and "Copy" not in f]
    for base in d_bases:
        prefix = base.split(".")[0]
        copy_candidates = [f for f in pdfs if prefix in f and "Copy" in f]
        if copy_candidates:
            # V1 = Copy, V2 = Base
            pairs.append((os.path.join(drawings_dir, copy_candidates[0]), os.path.join(drawings_dir, base)))

    print(f"Found {len(pairs)} pairs to process.")
    
    for v1, v2 in pairs:
        v1_name = os.path.basename(v1)
        v2_name = os.path.basename(v2)
        
        # Output name based on the base ID
        base_id = v2_name.replace(".PDF", "").replace(".pdf", "")
        out_path = os.path.join(output_base, f"audit_{base_id}.png")
        
        print(f"\n--- Processing: {base_id} ---")
        print(f"  V1: {v1_name}")
        print(f"  V2: {v2_name}")
        
        try:
            run_diff(v1, v2, out_path)
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    run_audit()
