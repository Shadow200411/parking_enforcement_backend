import re


PLATE_SANITIZE_PATTERN = re.compile(r"[^A-Z0-9]")


def normalize_registration_no(value: str) -> str:
    normalized = PLATE_SANITIZE_PATTERN.sub("", value.upper())
    return normalized.strip()
