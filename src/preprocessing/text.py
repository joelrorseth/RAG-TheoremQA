import re


def strip_excessive_whitespace(mystr: str) -> str:
    return re.sub('\s+', ' ', mystr).strip()
