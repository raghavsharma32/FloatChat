# api/utils/nl_router.py

from typing import Dict, Any

COL_SYNONYMS = {
    "temperature": {"temperature","temp","t","sst"},
    "salinity": {"salinity","salt","psu","s"},
}

def _asks_for_rows(q: str) -> bool:
    ql = q.lower()
    triggers = ["where", "lat", "lon", "long", "longitude", "latitude",
                "location", "coordinates", "which float", "float id",
                "position", "coord", "geo", "at what"]
    return any(t in ql for t in triggers)

def parse_nl_question(q: str) -> Dict[str, Any]:
    ql = q.lower()
    # operation
    if any(w in ql for w in ["highest","max","hottest","peak"]): op="max"
    elif any(w in ql for w in ["lowest","min","coldest"]): op="min"
    else: op="mean"

    # metric
    if "salin" in ql: value_col = "salinity"
    elif any(w in ql.split() for w in ["t","temp","temperature","sst"]): value_col = "temperature"
    else: value_col = "temperature"

    include_rows = _asks_for_rows(ql)
    return_cols = ["latitude","longitude","time"] if include_rows else []

    return {
        "op": op,
        "value_col": value_col,
        "include_rows": include_rows,   # 👈 NEW
        "return_cols": return_cols if include_rows else None,
        "filters": None,
    }
