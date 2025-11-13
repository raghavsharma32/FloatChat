# utils/intent.py
import re
from typing import Literal

Intent = Literal["count", "aggregate", "filter", "semantic"]

_COUNT_PAT = re.compile(r"\b(how many|count|number of|total)\b", re.I)
_AGG_PAT = re.compile(r"\b(min(imum)?|max(imum)?|avg|average|mean|median|sum)\b", re.I)
_FILTER_PAT = re.compile(r"\b(lat|lon|latitude|longitude|between|near|range|cycle|time|date|depth|pressure)\b", re.I)

def detect_intent(q: str) -> Intent:
    ql = q.lower().strip()
    if _COUNT_PAT.search(ql):
        return "count"
    if _AGG_PAT.search(ql):
        return "aggregate"
    if _FILTER_PAT.search(ql):
        return "filter"
    return "semantic"
