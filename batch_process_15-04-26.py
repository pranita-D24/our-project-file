import os
import glob
from pass1_added import run as run_diff

def process_batch():
    root_dir = r"15-04-26"
    output_dir = r"visuals\batch_15-04-26"
    os.makedirs(output_dir, exist_ok=True)

    # Find all PDFs recursively in the folder
    pdf_files = glob.glob(os.path.join(root_dir, "**", "*.pdf"), recursive=True)
    pdf_files.sort() # Ensure model15, model16, model17 order

    if len(pdf_files) < 2:
        print(f"Not enough PDF files found in {root_dir} (found {len(pdf_files)})")
        return

    print(f"Found {len(pdf_files)} PDF files. Processing sequential pairs...")

    # Pair them sequentially: (0,1), (1,2), etc.
    for i in range(len(pdf_files) - 1):
        v1_path = pdf_files[i]
        v2_path = pdf_files[i+1]
        
        v1_name = os.path.basename(v1_path)
        v2_name = os.path.basename(v2_path)
        
        # Output filename: e.g., diff_model15_to_model16.png
        out_name = f"diff_{v1_name.replace('.pdf', '')}_to_{v2_name.replace('.pdf', '')}.png"
        out_path = os.path.join(output_dir, out_name)
        
        print(f"\n--- Comparing {v1_name} vs {v2_name} ---")
        try:
            # We use the pass1_added.py logic (Pixel-based)
            run_diff(v1_path, v2_path, out_path)
        except Exception as e:
            print(f"Error processing {v1_name} vs {v2_name}: {e}")

    print(f"\nBatch processing complete. All reports saved to: {output_dir}")

if __name__ == "__main__":
    process_batch()
