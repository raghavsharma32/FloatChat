# api/utils/qa_agg.py
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

AGG_ALIASES = {
    "max": {"max","highest","largest","peak","hottest"},
    "min": {"min","lowest","smallest","coldest"},
    "mean": {"mean","avg","average"},
    "sum": {"sum","total"},
    "count": {"count","how many","number of"},
}

def _normalize_op(op: str) -> str:
    op = op.lower().strip()
    for canonical, aliases in AGG_ALIASES.items():
        if op in aliases:
            return canonical
    return op

def _apply_filters(df: pd.DataFrame, filters: Optional[Dict[str, Any]]) -> pd.DataFrame:
    if not filters:
        return df
    out = df
    for col, cond in filters.items():
        if isinstance(cond, dict):
            for k, v in cond.items():
                if k == "gt": out = out[out[col] > v]
                elif k == "lt": out = out[out[col] < v]
                elif k == "ge": out = out[out[col] >= v]
                elif k == "le": out = out[out[col] <= v]
                elif k == "eq": out = out[out[col] == v]
                elif k == "ne": out = out[out[col] != v]
                elif k == "in": out = out[out[col].isin(v)]
                elif k == "between": out = out[out[col].between(v[0], v[1], inclusive="both")]
        else:
            out = out[out[col] == cond]
    return out

def qa_aggregate(
    df: pd.DataFrame,
    value_col: str,
    op: str,
    *,
    include_rows: bool = False,            # 👈 NEW
    return_cols: Optional[List[str]] = None,
    filters: Optional[Dict[str, Any]] = None,
    tie_policy: str = "all",
    dropna: bool = True,
    units: Optional[str] = None,
    precision: Optional[int] = 3,
    id_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute aggregate and (optionally) return supporting rows if include_rows=True.
    """
    op = _normalize_op(op)
    work = _apply_filters(df, filters)
    if dropna:
        work = work[~work[value_col].isna()]

    if work.empty:
        return {
            "aggregate": None,
            "operation": op,
            "value_col": value_col,
            "units": units,
            "rows": [],
            "count": 0,
            "provenance": {"filters": filters or {}, "id_cols": id_cols or [], "tie_policy": tie_policy, "dropna": dropna},
            "notes": "No rows after filters.",
        }

    # Compute the aggregate value
    if op in {"max", "min"}:
        agg_val = getattr(work[value_col], op)()
    elif op in {"mean", "sum", "count"}:
        s = work[value_col]
        agg_val = getattr(s, op)() if op != "count" else s.count()
    else:
        raise ValueError(f"Unsupported operation: {op}")

    rows_out: List[Dict[str, Any]] = []
    if include_rows:
        # Only compute/return rows when explicitly asked
        if op in {"max", "min"}:
            rows = work[work[value_col] == agg_val]
            if tie_policy == "first":
                rows = rows.iloc[[0]]
        else:
            # For non-arg ops, return top contributors (up to 5)
            rows = work.sort_values(by=value_col, ascending=(op == "min")).head(5)

        keep_cols = set((return_cols or [])) | {value_col}
        if id_cols: keep_cols |= set(id_cols)
        # auto-include common geo cols if present and not already requested
        for c in ["lat", "latitude", "lon", "longitude"]:
            if c in work.columns:
                keep_cols.add(c)
        rows_out_df = rows.loc[:, [c for c in rows.columns if c in keep_cols]].copy()
        if precision is not None:
            for c in rows_out_df.select_dtypes(include=[np.number]).columns:
                rows_out_df[c] = rows_out_df[c].round(precision)
        rows_out = rows_out_df.to_dict(orient="records")

    result = {
        "aggregate": float(agg_val) if isinstance(agg_val, (int, float, np.floating)) else agg_val,
        "operation": op,
        "value_col": value_col,
        "units": units,
        "rows": rows_out,
        "count": int(len(rows_out)),
        "provenance": {"filters": filters or {}, "id_cols": id_cols or [], "tie_policy": tie_policy, "dropna": dropna},
        "notes": None,
    }
    return result
