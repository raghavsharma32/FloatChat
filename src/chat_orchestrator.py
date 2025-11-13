import os
import re
import pandas as pd
from typing import Dict, Any, Optional, Tuple

from utils.sql_tool import SQLTool
from src.float_rag import FloatChatRAG
from utils.db_connector import DBLoader
from exception.custom_exception import FloatChatException
from logger.custom_logger import CustomLogger


class ChatOrchestrator:
    """
    Orchestrates a hybrid chat pipeline:
    - Structured intent (counts, aggregates, filters) → SQLTool
    - Semantic/narrative queries → FAISS + RAG
    - SQL results are summarized by the LLM for clear NL answers
    """

    def __init__(self, session_id: str, db_url: str, faiss_dir: str):
        self.logger = CustomLogger().get_logger(__file__)
        self.session_id = session_id
        self.db = DBLoader(db_url)
        self.sql_tool = SQLTool(self.db.engine)
        self.faiss_dir = faiss_dir
        self.rag = FloatChatRAG(session_id)

    # ---------------------------
    # Setup / Routing
    # ---------------------------

    def _ensure_retriever_loaded(self):
        """Load FAISS retriever for this session if not already loaded."""
        path = os.path.join(self.faiss_dir, self.session_id)
        self.rag.load_retriever_from_faiss(path)

    def _is_structured_query(self, query: str) -> bool:
        """
        Heuristic: if query likely requires exact numbers/filters → SQL.
        """
        q = query.lower()
        keywords = [
            # filtering / geo / time / id
            "lat", "lon", "latitude", "longitude", "near", "between", "within",
            "time", "date", "from", "to", "since", "until", "range",
            "float_id", "cycle", "profiles",
            # numeric intents
            "how many", "count", "average", "avg", "mean", "median", "min", "max", "sum",
            # explicit numeric comparisons often imply SQL
            ">", "<", ">=", "<=", "=",
        ]
        return any(kw in q for kw in keywords)

    # ---------------------------
    # Entry point
    # ---------------------------

    def chat(self, query: str, include_rows: bool = False) -> Dict[str, Any]:
        """
        Routes to SQL, FAISS, or hybrid and returns:
        {
          "answer": str,
          "dataframe": Optional[list[dict]],
          "sources": list[str]
        }
        """
        try:
            self.logger.info(
                "🔎 Processing chat query", query=query, include_rows=include_rows
            )
            self._ensure_retriever_loaded()
            q = query.lower()

            # -------- Structured branch (SQL) --------
            if self._is_structured_query(q):
                self.logger.info("🗄️ Routed to SQLTool", query=query)

                # 1) COUNT style
                if "how many" in q or re.search(r"\bcount(s)?\b", q):
                    cnt = self.sql_tool.count_rows(self.session_id)
                    answer = f"There are {cnt} profiles in the database for this session."
                    return {"answer": answer, "dataframe": None, "sources": ["SQL: argo_profiles"]}

                # 2) Aggregate style (avg/min/max/sum/median)
                agg = self._detect_aggregate_op(q)
                if agg:
                    col = self._detect_column(q)

                    if include_rows:
                        agg_val, rows = self.sql_tool.aggregate_with_rows(
                            self.session_id, agg, col
                        )
                        if agg_val is None:
                            answer = f"I couldn’t compute the {agg} of {col} (no matching data)."
                            return {"answer": answer, "dataframe": None, "sources": ["SQL: argo_profiles"]}

                        try:
                            answer = f"The {agg} {col} is {float(agg_val):.3f}."
                        except Exception:
                            answer = f"The {agg} {col} is {agg_val}."

                        return {
                            "answer": answer,
                            "dataframe": rows if rows else None,
                            "sources": ["SQL: argo_profiles"],
                        }

                    else:
                        val = self.sql_tool.aggregate(self.session_id, agg, col)
                        if val is None:
                            answer = f"I couldn’t compute the {agg} of {col} (no matching data)."
                        else:
                            try:
                                answer = f"The {agg} {col} is {float(val):.3f}."
                            except Exception:
                                answer = f"The {agg} {col} is {val}."
                        return {"answer": answer, "dataframe": None, "sources": ["SQL: argo_profiles"]}

                # 3) Default SQL (bounded sample if we can't parse more)
                sql = self._nl_to_sql(query)
                df = self.sql_tool.run_query(sql)

                context = (
                    df.head(20).to_string(index=False)
                    if isinstance(df, pd.DataFrame) and not df.empty
                    else "No data found."
                )
                nl_prompt = f"User asked: {query}\nHere is the SQL result:\n{context}"
                answer = self.rag.invoke(nl_prompt)

                return {
                    "answer": answer,
                    "dataframe": (
                        df.to_dict(orient="records") if isinstance(df, pd.DataFrame) else None
                    ),
                    "sources": ["SQL: argo_profiles"],
                }

            # -------- Semantic branch (FAISS + RAG) --------
            self.logger.info("🧠 Routed to FAISS/RAG", query=query)
            answer = self.rag.invoke(query, include_rows=include_rows)
            return {"answer": answer, "dataframe": None, "sources": ["FAISS Index"]}

        except Exception as e:
            raise FloatChatException("Chat orchestration failed", e) from e

    # ---------------------------
    # NL → SQL (very basic stub)
    # ---------------------------

    def _nl_to_sql(self, query: str) -> str:
        q = query.lower()

        if "salinity" in q and "equator" in q and "march 2023" in q:
            return """
            SELECT cycle_number, time, lat, lon, depth, salinity
            FROM argo_profiles
            WHERE session_id = :sid
              AND lat BETWEEN -5 AND 5
              AND time >= '2023-03-01' AND time < '2023-04-01'
            ORDER BY time, depth;
            """.replace(":sid", f"'{self.session_id}'")

        return f"""
        SELECT cycle_number, time, lat, lon, depth, temperature, salinity
        FROM argo_profiles
        WHERE session_id = '{self.session_id}'
        ORDER BY time DESC
        LIMIT 50;
        """

    # ---------------------------
    # Helpers: aggregate + column detection
    # ---------------------------

    def _detect_aggregate_op(self, q: str) -> Optional[str]:
        q = q.lower()
        patterns = [
            (r"\b(average|avg|mean)\b", "avg"),
            (r"\bmedian\b", "median"),
            (r"\bminimum\b|\bmin\b|\blowest\b|\bsmallest\b", "min"),
            (r"\bmaximum\b|\bmax\b|\bhighest\b|\blargest\b", "max"),
            (r"\bsum\b|\btotal\b", "sum"),
        ]
        for pat, op in patterns:
            if re.search(pat, q):
                return op
        return None

    def _detect_column(self, q: str) -> str:
        q = q.lower().strip()
        column_map: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
            ("temperature", (
                r"\btemperature\b", r"\btemp\b", r"\btemps\b",
                r"\bpotential temperature\b", r"\bptemp\b", r"\bt\b",
            )),
            ("salinity", (
                r"\bsalinity\b", r"\bsalin\b", r"\bsalt\b", r"\bpsu\b",
                r"\bpractical salinity\b", r"\bsp\b",
            )),
            ("depth", (
                r"\bdepth\b", r"\bpressure\b", r"\bpress\b", r"\bp\b",
                r"\bdepth\s*(m|meters|metres)?\b", r"\bdbar\b",
            )),
            ("lat", (r"\blat\b", r"\blatitude\b")),
            ("lon", (r"\blon\b", r"\blongitude\b")),
        )
        for col, patterns in column_map:
            for pat in patterns:
                if re.search(pat, q):
                    return col

        if re.search(r"\b(salt|psu|salin)\b", q):
            return "salinity"
        if re.search(r"\b(temp|temperature|ptemp)\b", q):
            return "temperature"
        if re.search(r"\b(depth|pressure|dbar)\b", q):
            return "depth"

        return "temperature"
