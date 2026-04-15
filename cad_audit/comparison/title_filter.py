def is_administrative(span: dict) -> bool:
    """Returns True if the span is in the title block zone (bottom 20%)."""
    page_h = span["page_height"]
    cy = span["centroid"][1]
    # Bottom 20% = title block
    return cy > (page_h * 0.80)

def filter_structural(spans: list) -> list:
    """Preserves only spans that are NOT in administrative zones."""
    return [s for s in spans if not is_administrative(s)]
