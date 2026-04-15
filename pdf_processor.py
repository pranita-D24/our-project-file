# pdf_processor.py
# Industry Grade PDF Processing Module
# Handles PDF upload, validation, and conversion

import os
import fitz  # pymupdf
import logging
import hashlib
import shutil
from datetime import datetime
from PIL import Image
import json
from config import (
    UPLOAD_DIR, IMAGE_DIR, POPPLER_PATH,
    IMAGE_DPI, TARGET_SIZE
)

# ══════════════════════════════════════
# LOGGING
# ══════════════════════════════════════
logger = logging.getLogger(__name__)

# ══════════════════════════════════════
# PDF PROCESSOR CLASS
# ══════════════════════════════════════
class PDFProcessor:

    def __init__(self):
        self.upload_dir = UPLOAD_DIR
        self.image_dir  = IMAGE_DIR
        logger.info("PDFProcessor initialized")

    # ══════════════════════════════════
    # VALIDATION
    # ══════════════════════════════════
    def validate_pdf(self, pdf_path):
        """
        Validates PDF file before processing
        Checks: exists, readable, not corrupted,
                not empty, is actual PDF
        """
        errors = []

        # Check file exists
        if not os.path.exists(pdf_path):
            errors.append(f"File not found: {pdf_path}")
            return False, errors

        # Check file size
        size = os.path.getsize(pdf_path)
        if size == 0:
            errors.append("File is empty")
            return False, errors

        if size > 500 * 1024 * 1024:  # 500MB limit
            errors.append("File too large (max 500MB)")
            return False, errors

        # Check PDF header
        try:
            with open(pdf_path, "rb") as f:
                header = f.read(5)
                if header != b"%PDF-":
                    errors.append("Not a valid PDF file")
                    return False, errors
        except Exception as e:
            errors.append(f"Cannot read file: {e}")
            return False, errors

        # Check PDF is not corrupted
        try:
            doc = fitz.open(pdf_path)
            if doc.page_count == 0:
                errors.append("PDF has no pages")
                return False, errors
            doc.close()
        except Exception as e:
            errors.append(f"PDF corrupted: {e}")
            return False, errors

        return True, []

    # ══════════════════════════════════
    # FILE HASH
    # ══════════════════════════════════
    def get_file_hash(self, pdf_path):
        """
        Generate MD5 hash of PDF
        Used to detect duplicate uploads
        """
        hasher = hashlib.md5()
        try:
            with open(pdf_path, "rb") as f:
                while chunk := f.read(8192):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"Hash error: {e}")
            return None

    # ══════════════════════════════════
    # GET PDF INFO
    # ══════════════════════════════════
    def get_pdf_info(self, pdf_path):
        """
        Extract PDF metadata and information
        Returns page count, dimensions, metadata
        """
        try:
            doc  = fitz.open(pdf_path)
            info = {
                "page_count"  : doc.page_count,
                "file_size"   : os.path.getsize(pdf_path),
                "file_hash"   : self.get_file_hash(pdf_path),
                "pages"       : [],
                "metadata"    : doc.metadata or {}
            }

            for i, page in enumerate(doc):
                rect = page.rect
                info["pages"].append({
                    "page_no" : i + 1,
                    "width"   : rect.width,
                    "height"  : rect.height
                })

            doc.close()
            return info

        except Exception as e:
            logger.error(f"PDF info error: {e}")
            raise

    # ══════════════════════════════════
    # SAVE PDF
    # ══════════════════════════════════
    def save_pdf(self, source_path, drawing_id,
                 version_no):
        """
        Save uploaded PDF to organized folder structure
        uploads/{drawing_id}/{version_no}.pdf
        """
        try:
            # Create folder for this drawing
            drawing_folder = os.path.join(
                self.upload_dir, drawing_id)
            os.makedirs(drawing_folder, exist_ok=True)

            # Destination path
            dest_filename = f"{version_no}.pdf"
            dest_path     = os.path.join(
                drawing_folder, dest_filename)

            # Copy file
            shutil.copy2(source_path, dest_path)
            logger.info(
                f"PDF saved: {dest_path}")
            return dest_path

        except Exception as e:
            logger.error(f"PDF save error: {e}")
            raise

    # ══════════════════════════════════
    # CONVERT PDF TO IMAGES
    # ══════════════════════════════════
    def convert_to_images(self, pdf_path, drawing_id,
                          version_no):
        """
        Convert PDF pages to high quality images
        Uses PyMuPDF for best quality
        Saves to images/{drawing_id}/{version_no}/
        Returns list of saved image paths
        """
        image_paths = []

        try:
            # Create output folder
            output_folder = os.path.join(
                self.image_dir, drawing_id, version_no)
            os.makedirs(output_folder, exist_ok=True)

            # Open PDF
            doc    = fitz.open(pdf_path)
            matrix = fitz.Matrix(
                IMAGE_DPI / 72, IMAGE_DPI / 72)

            logger.info(
                f"Converting {doc.page_count} pages "
                f"from {os.path.basename(pdf_path)}")

            for page_num in range(doc.page_count):
                page      = doc[page_num]
                pixmap    = page.get_pixmap(matrix=matrix)

                # Convert to PIL Image
                img = Image.frombytes(
                    "RGB",
                    [pixmap.width, pixmap.height],
                    pixmap.samples
                )

                # Resize to standard size
                img = img.resize(
                    TARGET_SIZE, Image.LANCZOS)

                # Save image
                img_name = f"page_{page_num + 1}.jpg"
                img_path = os.path.join(
                    output_folder, img_name)
                img.save(img_path, "JPEG",
                         quality=95,
                         optimize=True)

                image_paths.append(img_path)
                logger.info(
                    f"  Page {page_num + 1} → {img_name}")

            doc.close()
            logger.info(
                f"Conversion complete: "
                f"{len(image_paths)} images created")
            return image_paths

        except Exception as e:
            logger.error(
                f"PDF conversion error: {e}")
            raise

    # ══════════════════════════════════
    # FULL PROCESS
    # ══════════════════════════════════
    def process_pdf(self, source_path, drawing_id,
                    version_no):
        """
        Complete PDF processing pipeline:
        1. Validate PDF
        2. Get PDF info
        3. Save PDF to system
        4. Convert to images
        Returns complete processing result
        """
        result = {
            "success"     : False,
            "drawing_id"  : drawing_id,
            "version_no"  : version_no,
            "pdf_path"    : None,
            "image_paths" : [],
            "page_count"  : 0,
            "pdf_info"    : {},
            "errors"      : [],
            "processed_at": datetime.now().isoformat()
        }

        try:
            # Step 1 — Validate
            logger.info(
                f"Processing PDF: "
                f"{os.path.basename(source_path)}")
            valid, errors = self.validate_pdf(source_path)
            if not valid:
                result["errors"] = errors
                logger.error(
                    f"Validation failed: {errors}")
                return result

            # Step 2 — Get info
            pdf_info          = self.get_pdf_info(
                source_path)
            result["pdf_info"]    = pdf_info
            result["page_count"]  = pdf_info["page_count"]

            # Step 3 — Save PDF
            saved_path        = self.save_pdf(
                source_path, drawing_id, version_no)
            result["pdf_path"] = saved_path

            # Step 4 — Convert to images
            image_paths       = self.convert_to_images(
                saved_path, drawing_id, version_no)
            result["image_paths"] = image_paths

            result["success"] = True
            logger.info(
                f"PDF processing complete ✅ "
                f"{len(image_paths)} images")
            return result

        except Exception as e:
            result["errors"].append(str(e))
            logger.error(
                f"PDF processing failed: {e}")
            return result

    # ══════════════════════════════════
    # GET IMAGE PATHS
    # ══════════════════════════════════
    def get_image_paths(self, drawing_id, version_no):
        """
        Get all image paths for a drawing version
        """
        folder = os.path.join(
            self.image_dir, drawing_id, version_no)

        if not os.path.exists(folder):
            return []

        images = sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.endswith(".jpg")
        ])
        return images


# ══════════════════════════════════════
# TEST
# ══════════════════════════════════════
if __name__ == "__main__":

    print("Testing PDFProcessor...")
    print("=" * 50)

    processor  = PDFProcessor()
    
    # ── Automatically find all PDFs in Drawings/ folder ──
    drawings_folder = "Drawings"
    
    if not os.path.exists(drawings_folder):
        print(f"Drawings folder not found!")
    else:
        # Get all PDF files automatically
        pdf_files = [
            f for f in os.listdir(drawings_folder)
            if f.upper().endswith('.PDF')
        ]

        print(f"Total PDFs found : {len(pdf_files)}")
        print()

        success_count = 0
        failed_count  = 0

        for i, pdf_file in enumerate(pdf_files):
            pdf_path   = os.path.join(
                drawings_folder, pdf_file)
            
            # Use filename without extension as drawing ID
            drawing_id = os.path.splitext(pdf_file)[0]
            drawing_id = drawing_id.replace(" ", "_")
            drawing_id = drawing_id.replace("-", "_")
            
            # Detect version from name
            if "Copy" in pdf_file:
                version_no = "v2"
            else:
                version_no = "v1"

            print(f"[{i+1}/{len(pdf_files)}] "
                  f"Processing: {pdf_file}")
            print(f"  Drawing ID : {drawing_id}")
            print(f"  Version    : {version_no}")

            result = processor.process_pdf(
                source_path = pdf_path,
                drawing_id  = drawing_id,
                version_no  = version_no
            )

            if result["success"]:
                print(f"  Pages      : {result['page_count']}")
                print(f"  Images     : "
                      f"{len(result['image_paths'])}")
                print(f"  Status     : ✅ Done")
                success_count += 1
            else:
                print(f"  Status     : ❌ Failed")
                for err in result["errors"]:
                    print(f"  Error      : {err}")
                failed_count += 1
            print()

        print("=" * 50)
        print(f"SUMMARY")
        print("=" * 50)
        print(f"Total    : {len(pdf_files)}")
        print(f"Success  : {success_count} ✅")
        print(f"Failed   : {failed_count} ❌")
        print()
        print("Check images/ folder for converted images")
        print("=" * 50)