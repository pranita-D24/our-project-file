# reasoning_engine.py
# Stage 7 — Human Intelligence Reasoning Layer
# Translates geometric diffs into "Mechanical Stories"

import logging
import json
import os
from typing import List, Dict, Any, Optional
from knowledge_manager import KnowledgeManager

logger = logging.getLogger(__name__)

class ReasoningEngine:
    """
    Acts as the 'Senior Engineer' brain.
    Uses LLMs to interpret the intent behind drawing changes.
    """

    INTENT_PROMPT = """You are a Senior Mechanical Engineer reviewing a drawing revision.
Analyze the following list of geometric changes and the overall drawing context.
Determine the likely 'Mechanical Intent' behind these changes.

CHANGES:
{changes_summary}

DESIGN STANDARDS & RULES:
{design_standards}

Provide your analysis in JSON:
{{
  "intent_hypothesis": "Short description of the guessed goal (e.g., 'Pressure upgrade')",
  "reasoning_logic": "Why you think this is the goal based on the standards",
  "confidence": 0.0-1.0
}}
"""

    VERDICT_PROMPT = """You are finalizing an Engineering Audit Memo.
Based on the following intent hypothesis and the detailed evidence, write a human-like summary.

INTENT: {intent}
EVIDENCE: {evidence}

The memo should:
1. State the primary reason for the revision.
2. Confirm if the geometric changes support this intent.
3. Call out any 'suspicious' or 'unexplained' changes.
4. Be professional and technical.
"""

    def __init__(self, model_name: str = "claude-3-5-sonnet"):
        self.model_name = model_name
        self.client = None
        self.knowledge_base = [] # Internal list
        self.km = KnowledgeManager()
        self._init_client()

    def _init_client(self):
        """Initialize appropriate AI client."""
        try:
            if "claude" in self.model_name.lower():
                import anthropic
                self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            # Gemini support can be added here
        except Exception as e:
            logger.error(f"ReasoningEngine client init error: {e}")

    def ingest_knowledge(self, pdf_path: str):
        """Ingests a Design Standard PDF into the engine's memory."""
        logger.info(f"Ingesting knowledge from {pdf_path}...")
        # Implementation for PDF text extraction would go here
        self.knowledge_base.append({"source": pdf_path, "status": "loaded"})

    def _call_llm(self, prompt: str, system_msg: str = "You are an expert engineering drawing reviewer.") -> str:
        """Private helper to call the LLM."""
        if not self.client:
            return "AI Analysis unavailable (API key not set)."
        
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=1000,
                system=system_msg,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"LLM call error: {e}")
            return f"Error during analysis: {e}"

    def analyze_intent(self, changes: List[Dict[str, Any]], drawing_standard: str = "UNKNOWN") -> Dict[str, Any]:
        """Phase 1: Generate Intent Hypothesis."""
        summary = self._summarize_changes(changes)
        standards = self.km.get_contextual_rules(drawing_standard)
        
        prompt = self.INTENT_PROMPT.format(
            changes_summary=summary,
            design_standards=standards
        )
        
        logger.info("Generating intent hypothesis via LLM with standards context...")
        raw_response = self._call_llm(prompt)
        
        try:
            # Attempt to parse JSON from the response
            start = raw_response.find("{")
            end = raw_response.rfind("}") + 1
            if start != -1 and end != -1:
                data = json.loads(raw_response[start:end])
                return data
        except Exception:
            logger.warning("Failed to parse JSON from intent response, using raw text.")
            
        return {
            "intent_hypothesis": raw_response[:100],
            "reasoning_logic": raw_response,
            "confidence": 0.5
        }

    def verify_changes(self, hypothesis: str, changes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Phase 2: Active Verification."""
        logger.info(f"Verifying evidence for hypothesis: {hypothesis}")
        verified_evidence = []
        for ch in changes:
            # Here we would use the LLM to verify each change
            ch["supported_by_intent"] = True
            verified_evidence.append(ch)
        return verified_evidence

    def generate_narrative(self, hypothesis: Dict[str, Any], evidence: List[Dict[str, Any]]) -> str:
        """Phase 3: Produce the final Mechanical Story."""
        logger.info("Writing final narrative verdict via LLM...")
        
        # Prepare evidence summary for the prompt
        evidence_summary = self._summarize_changes(evidence)
        prompt = self.VERDICT_PROMPT.format(
            intent=hypothesis.get("intent_hypothesis", "Unknown"),
            evidence=evidence_summary
        )
        
        return self._call_llm(prompt)

    def _summarize_changes(self, changes: List[Dict[str, Any]]) -> str:
        """Helper to create a text summary for the LLM."""
        if not changes:
            return "No changes detected."
        
        summary = []
        for ch in changes:
            summary.append(f"- {ch.get('status', 'MODIFIED')} {ch.get('type', 'element')} at {ch.get('centroid', 'unknown')}")
        return "\n".join(summary)

    def run_full_audit(self, changes: List[Dict[str, Any]], drawing_standard: str = "UNKNOWN") -> Dict[str, Any]:
        """The main entry point for Stage 7."""
        if not changes:
            return {"verdict": "No changes to analyze.", "story": ""}

        hypothesis = self.analyze_intent(changes, drawing_standard)
        verified = self.verify_changes(hypothesis["intent_hypothesis"], changes)
        story = self.generate_narrative(hypothesis, verified)

        return {
            "hypothesis": hypothesis,
            "evidence": verified,
            "mechanical_story": story
        }

if __name__ == "__main__":
    # Test Mock Data
    test_changes = [
        {"status": "ADDED", "type": "circle-cluster", "centroid": [100, 200]},
        {"status": "REMOVED", "type": "line-cluster", "centroid": [150, 250]},
        {"status": "RESIZED", "type": "rect-cluster", "centroid": [300, 400], "area_change": 15.5}
    ]
    
    engine = ReasoningEngine()
    print("\n--- STAGE 7 AUDIT TEST ---")
    result = engine.run_full_audit(test_changes)
    print(f"Hypothesis: {result['hypothesis']['intent_hypothesis']}")
    print(f"Narrative:\n{result['mechanical_story']}")
    print("--- TEST COMPLETE ---")
