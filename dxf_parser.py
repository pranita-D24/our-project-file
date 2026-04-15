# dxf_parser.py — DXF entity diff fast-path
# Uses ezdxf to compare entities between two DXF files without rasterising.
# Calls ingest() to prime the profile cache for both files before diffing.

import ezdxf
import logging
from comparator import CompareResult

logger = logging.getLogger(__name__)


def parse_dxf_entities(path):
    doc = ezdxf.readfile(path)
    msp = doc.modelspace()
    entities = []
    for e in msp:
        if e.dxftype() in ('LINE', 'CIRCLE', 'ARC', 'LWPOLYLINE', 'TEXT', 'MTEXT'):
            entities.append({
                "type": e.dxftype(),
                "layer": e.dxf.layer,
                "handle": e.dxf.handle,
            })
    return entities


def _hash_entity(e):
    return f"{e['type']}_{e['layer']}"


def diff_dxf(p1, p2) -> CompareResult:
    logger.info("Running DXF fast-pass diff")

    # Prime the profile cache for both files via the Stage 1 pipeline
    try:
        from ingestion import ingest
        ingest(p1)
        ingest(p2)
    except Exception as e:
        logger.warning(f"ingest() cache priming skipped: {e}")

    e1 = parse_dxf_entities(p1)
    e2 = parse_dxf_entities(p2)

    h1 = [_hash_entity(e) for e in e1]
    h2 = [_hash_entity(e) for e in e2]

    c_added   = 0
    c_removed = 0
    c_matched = 0

    h2_copy = h2[:]
    for h in h1:
        if h in h2_copy:
            c_matched += 1
            h2_copy.remove(h)
        else:
            c_removed += 1

    c_added = len(h2_copy)

    res = CompareResult(verdict="DXF_FAST_PASS", similarity=100.0)

    # Use a plain dict for processing_info — ProcessingTiming doesn't exist
    res.processing_info = {
        "added":   c_added,
        "removed": c_removed,
        "moved":   0,
        "resized": 0,
        "changed": 0,
    }

    if c_added > 0 or c_removed > 0:
        res.verdict    = "MAJOR CHANGES"
        res.similarity = round(c_matched / (c_matched + c_removed + 1e-6) * 100, 2)
    else:
        res.verdict    = "IDENTICAL / VERY SIMILAR"
        res.similarity = 100.0

    return res
