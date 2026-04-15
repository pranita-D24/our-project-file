# knowledge_manager.py
# Stage 8 — Domain Knowledge Ingestion
# Extracts and indexes engineering rules from PDFs

import os
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class KnowledgeManager:
    """
    Manages the ingestion and retrieval of engineering standards.
    """
    
    KNOWLEDGE_DIR = "knowledge_base"
    RULES_FILE = os.path.join(KNOWLEDGE_DIR, "rules.json")

    def __init__(self):
        if not os.path.exists(self.KNOWLEDGE_DIR):
            os.makedirs(self.KNOWLEDGE_DIR)
        self.rules = self._load_rules()

    def _load_rules(self) -> List[Dict[str, Any]]:
        """Loads existing rules from the knowledge base."""
        if os.path.exists(self.RULES_FILE):
            try:
                with open(self.RULES_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading rules: {e}")
        return []

    def ingest_standard(self, file_path: str):
        """
        Extracts rules from PDF or TXT files.
        """
        logger.info(f"Ingesting standard: {file_path}")
        try:
            ext = os.path.splitext(file_path)[1].lower()
            if ext == ".pdf":
                import fitz
                doc = fitz.open(file_path)
                full_text = ""
                for page in doc:
                    full_text += page.get_text()
            elif ext == ".txt":
                with open(file_path, "r", encoding="utf-8") as f:
                    full_text = f.read()
            else:
                logger.error(f"Unsupported file type: {ext}")
                return False
            
            # Save rule record
            new_rule = {
                "source": os.path.basename(file_path),
                "summary": "Engineering Standard Ingested",
                "full_text": full_text
            }
            
            self.rules.append(new_rule)
            self._save_rules()
            return True
        except Exception as e:
            logger.error(f"Standard ingestion failed: {e}")
            return False

    def _save_rules(self):
        """Saves current rules to the JSON index."""
        try:
            with open(self.RULES_FILE, "w", encoding="utf-8") as f:
                json.dump(self.rules, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving rules: {e}")

    def get_contextual_rules(self, drawing_standard: str) -> str:
        """
        Retrieves rules relevant to the current drawing's standard.
        """
        relevant = [r["full_text"] for r in self.rules if drawing_standard.upper() in r["source"].upper() or drawing_standard == "UNKNOWN"]
        if not relevant:
            return "No specific design standards were found for this drawing type."
        
        return "\n---\n".join(relevant)
