import re
from pathlib import PurePosixPath

INVALID_PATH_CHARS = re.compile(r"[\0]")

def sanitize_path(path: str) -> str:
    if INVALID_PATH_CHARS.search(path):
        raise ValueError("Invalid characters in path")
    p = PurePosixPath(path)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError("Path must be relative and must not contain traversal segments")
    normalized = str(p)
    if normalized.startswith("./"):
        normalized = normalized[2:]
    if normalized == "":
        raise ValueError("Empty path after normalization")
    return normalized
