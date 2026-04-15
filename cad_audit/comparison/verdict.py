def compute_verdict(added: int, removed: int) -> str:
    """Tiered verdict logic based on structural and dimensional change counts."""
    total = added + removed
    if total == 0:
        return "NO CHANGE"
    elif total <= 10:
        return "MINOR CHANGES"
    elif total <= 30:
        return "MODERATE CHANGES"
    else:
        return "MAJOR CHANGES"
