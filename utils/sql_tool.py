# utils/sql_tool.py
import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from typing import Optional, Dict, Any, List, Tuple

from exception.custom_exception import FloatChatException
from logger.custom_logger import CustomLogger

# whitelist of safe columns to prevent SQL injection
SAFE_COLS = {
    "float_id", "cycle_number", "time", "lat", "lon", "depth",
    "temperature", "salinity", "qc_temp", "qc_salin", "session_id"
}

# sensible default columns to show when returning rows
DEFAULT_ROW_COLS = [
    "float_id", "cycle_number", "time", "lat", "lon", "depth",
    "temperature", "salinity"
]


class SQLTool:
    """
    Utility class for safe SQL querying on PostgreSQL.
    - Allows only SELECT queries.
    - Aggregations with optional supporting rows.
    """

    def __init__(self, engine):
        self.logger = CustomLogger().get_logger(__file__)
        self.engine = engine

    # ---------- General query ----------
    def run_query(self, sql: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Run a SELECT query and return results as DataFrame.
        Rejects non-SELECT queries for safety.
        """
        try:
            sql_clean = sql.strip().lower()
            if not sql_clean.startswith("select"):
                raise FloatChatException("❌ Only SELECT queries are allowed", sql)

            self.logger.info("📊 Executing SQL query", sql=sql, params=params or {})

            with self.engine.begin() as conn:   # ✅ transactional, safe
                df = pd.read_sql(text(sql), conn, params=params or {})

            self.logger.info(
                "✅ SQL query executed",
                rows=len(df),
                columns=list(df.columns),
            )
            return df

        except SQLAlchemyError as e:
            self.logger.error("❌ SQLAlchemy error", error=str(e), sql=sql)
            raise FloatChatException("SQL execution failed", e) from e
        except Exception as e:
            self.logger.error("❌ Unexpected SQL error", error=str(e), sql=sql)
            raise FloatChatException("SQL tool encountered an error", e) from e

    # ---------- Aggregations (scalar) ----------
    def count_rows(self, session_id: str, where_sql: str = "", params: Optional[Dict[str, Any]] = None) -> int:
        """
        Count rows safely for a given session_id + optional WHERE.
        """
        try:
            base = "SELECT COUNT(*) AS cnt FROM argo_profiles WHERE session_id = :sid"
            sql = f"{base} {('AND ' + where_sql) if where_sql else ''};"
            with self.engine.begin() as conn:
                res = conn.execute(text(sql), {"sid": session_id, **(params or {})})
                row = res.fetchone()
            return int(row[0]) if row else 0
        except Exception as e:
            raise FloatChatException("COUNT query failed", e)

    def aggregate(
        self,
        session_id: str,
        op: str,
        column: str,
        where_sql: str = "",
        params: Optional[Dict[str, Any]] = None
    ) -> Optional[float]:
        """
        Run aggregation like MIN/MAX/AVG/SUM for a safe column and return a scalar.
        """
        op_sql = self._map_op(op)
        self._validate_column(column)

        try:
            base = f"SELECT {op_sql}({column}) AS val FROM argo_profiles WHERE session_id = :sid"
            sql = f"{base} {('AND ' + where_sql) if where_sql else ''};"
            with self.engine.begin() as conn:
                res = conn.execute(text(sql), {"sid": session_id, **(params or {})})
                row = res.fetchone()
            return float(row[0]) if row and row[0] is not None else None
        except Exception as e:
            raise FloatChatException("Aggregate query failed", e)

    # ---------- Aggregations (with rows) ----------
    def aggregate_with_rows(
        self,
        session_id: str,
        op: str,
        column: str,
        *,
        where_sql: str = "",
        params: Optional[Dict[str, Any]] = None,
        top_n: int = 5,
        tie_policy: str = "all",        # "all" or "first"
        return_cols: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[List[Dict[str, Any]]]]:
        """
        Compute an aggregate AND optionally return supporting rows.
        - For MIN/MAX: return all rows where column equals the aggregate (ties respected).
        - For AVG/SUM/COUNT: return top-N rows by the column as "contributors" (optional).
          If you don't want contributors for non-arg ops, set top_n=0.

        Returns: (aggregate_value or None, rows as list[dict] or None)
        """
        op_sql = self._map_op(op)
        self._validate_column(column)
        params = params or {}

        try:
            # 1) Compute the aggregate value
            base_agg = f"SELECT {op_sql}({column}) AS val FROM argo_profiles WHERE session_id = :sid"
            sql_agg = f"{base_agg} {('AND ' + where_sql) if where_sql else ''};"
            with self.engine.begin() as conn:
                res = conn.execute(text(sql_agg), {"sid": session_id, **params})
                row = res.fetchone()
                agg_val = row[0] if row else None

            if agg_val is None:
                return None, None

            # 2) Collect supporting rows
            cols = self._build_return_cols(column, return_cols)

            if op_sql in ("MIN", "MAX"):
                # Equality match for ties
                base_rows = f"""
                    SELECT {', '.join(cols)}
                    FROM argo_profiles
                    WHERE session_id = :sid
                      AND {column} = :agg_val
                """
                sql_rows = f"{base_rows} {('AND ' + where_sql) if where_sql else ''}"
                if tie_policy == "first":
                    sql_rows += " LIMIT 1"
                sql_rows += ";"

                with self.engine.begin() as conn:
                    df = pd.read_sql(
                        text(sql_rows),
                        conn,
                        params={"sid": session_id, "agg_val": agg_val, **params}
                    )

                rows = df.to_dict(orient="records") if not df.empty else []
                return (float(agg_val), rows)

            else:
                # AVG/SUM/COUNT contributors (optional)
                if top_n <= 0:
                    return (float(agg_val), None)

                order = "DESC"  # largest contributors first
                base_rows = f"""
                    SELECT {', '.join(cols)}
                    FROM argo_profiles
                    WHERE session_id = :sid
                      {('AND ' + where_sql) if where_sql else ''}
                    ORDER BY {column} {order}
                    LIMIT :top_n
                """
                with self.engine.begin() as conn:
                    df = pd.read_sql(
                        text(base_rows),
                        conn,
                        params={"sid": session_id, "top_n": top_n, **params}
                    )

                rows = df.to_dict(orient="records") if not df.empty else []
                return (float(agg_val), rows if rows else None)

        except Exception as e:
            raise FloatChatException("Aggregate-with-rows query failed", e)

    # ---------- Helpers ----------
    def build_where_between(self, col: str, a_param: str, b_param: str) -> Tuple[str, List[str]]:
        """
        Return safe WHERE clause for BETWEEN queries.
        Example:
            where_sql, keys = sql_tool.build_where_between("lat", "lat_a", "lat_b")
        """
        self._validate_column(col)
        return f"{col} BETWEEN :{a_param} AND :{b_param}", [a_param, b_param]

    def _validate_column(self, col: str):
        if col not in SAFE_COLS:
            raise FloatChatException(f"Unsafe/unknown column: {col}")

    def _map_op(self, op: str) -> str:
        op_map = {
            "min": "MIN",
            "max": "MAX",
            "avg": "AVG",
            "average": "AVG",
            "mean": "AVG",
            "sum": "SUM",
            "count": "COUNT",
            "median": None,  # not supported natively; implement via percentile if needed
        }
        sql_op = op_map.get(op.lower())
        if not sql_op:
            raise FloatChatException(f"Unsupported aggregate op: {op}")
        return sql_op

    def _build_return_cols(self, value_col: str, return_cols: Optional[List[str]]) -> List[str]:
        """
        Build a safe, de-duplicated list of columns to select when returning rows.
        Always includes the aggregated value column and common geo/id columns if available.
        """
        cols: List[str] = []
        wanted = list(return_cols or []) + DEFAULT_ROW_COLS + [value_col]

        seen = set()
        for c in wanted:
            if c in SAFE_COLS and c not in seen:
                cols.append(c)
                seen.add(c)

        # Guarantee at least the value column is present
        if value_col not in cols and value_col in SAFE_COLS:
            cols.append(value_col)

        return cols
