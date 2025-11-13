# utils/db_connector.py

import os
import pandas as pd
import uuid
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

from exception.custom_exception import FloatChatException
from logger.custom_logger import CustomLogger


class DBLoader:
    def __init__(self, db_url: str = None):
        load_dotenv()  # Load env vars

        self.logger = CustomLogger().get_logger(__file__)
        self.db_url = db_url or os.getenv(
            "DATABASE_URL", "postgresql://postgres:password@localhost:5432/floatchat"
        )

        try:
            self.engine = create_engine(self.db_url)
            self.logger.info("✅ Connected to PostgreSQL", db_url=self.db_url)
        except SQLAlchemyError as e:
            raise FloatChatException("Failed to connect to PostgreSQL", e)

    def create_table(self):
        """
        Create the argo_profiles table if it doesn't exist.
        Includes session_id for session-scoped data handling.
        """
        try:
            create_stmt = """
            CREATE TABLE IF NOT EXISTS argo_profiles (
                id SERIAL PRIMARY KEY,
                session_id UUID NOT NULL,
                float_id VARCHAR NOT NULL,
                cycle_number INT,
                time TIMESTAMP,
                lat DOUBLE PRECISION,
                lon DOUBLE PRECISION,
                depth DOUBLE PRECISION,
                temperature DOUBLE PRECISION,
                salinity DOUBLE PRECISION,
                qc_temp VARCHAR,
                qc_salin VARCHAR,
                created_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(session_id, float_id, cycle_number, time, depth)
            );
            """
            with self.engine.begin() as conn:
                conn.execute(text(create_stmt))
            self.logger.info("✅ Table argo_profiles created or already exists")
        except SQLAlchemyError as e:
            raise FloatChatException("Failed to create table", e)

    def new_session(self) -> str:
        """Generate a new session ID (UUID)."""
        sid = str(uuid.uuid4())
        self.logger.info("🆕 New session created", session_id=sid)
        return sid

    def insert_profiles(self, df: pd.DataFrame, session_id: str):
    
        try:
            expected_cols = [
                "float_id", "cycle_number", "time", "lat", "lon",
                "depth", "temperature", "salinity", "qc_temp", "qc_salin"
            ]

            df_subset = df[[c for c in expected_cols if c in df.columns]].copy()
            df_subset["session_id"] = session_id

            if df_subset.empty:
                self.logger.warning("⚠️ No valid columns found to insert", session_id=session_id)
                return

            # Drop exact duplicates in DataFrame
            before = len(df_subset)
            df_subset = df_subset.drop_duplicates()
            after = len(df_subset)
            if before > after:
                self.logger.info(f"🗑️ Dropped {before - after} true duplicates before DB insert")

            # Use raw SQL for conflict handling
            insert_stmt = """
            INSERT INTO argo_profiles 
            (float_id, cycle_number, time, lat, lon, depth, temperature, salinity, qc_temp, qc_salin, session_id)
            VALUES (:float_id, :cycle_number, :time, :lat, :lon, :depth, :temperature, :salinity, :qc_temp, :qc_salin, :session_id)
            ON CONFLICT (session_id, float_id, cycle_number, time, depth) DO NOTHING;
            """

            with self.engine.begin() as conn:
                conn.execute(text(insert_stmt), df_subset.to_dict(orient="records"))

            self.logger.info("✅ Inserted profiles", rows=len(df_subset), session_id=session_id)

        except Exception as e:
            raise FloatChatException("Failed to insert profiles into PostgreSQL", e)


    def query_profiles(self, session_id: str, limit: int = 100):
        """Fetch rows for a given session."""
        try:
            query = """
            SELECT * FROM argo_profiles
            WHERE session_id = :sid
            ORDER BY time
            LIMIT :limit;
            """
            with self.engine.begin() as conn:
                result = conn.execute(text(query), {"sid": session_id, "limit": limit})
                rows = result.fetchall()

            return [dict(r._mapping) for r in rows]
        except SQLAlchemyError as e:
            raise FloatChatException("Failed to query profiles", e)

    def clear_session(self, session_id: str):
        """Delete rows for a specific session."""
        try:
            with self.engine.begin() as conn:
                conn.execute(text("DELETE FROM argo_profiles WHERE session_id = :sid;"), {"sid": session_id})
            self.logger.info("🗑️ Cleared session data", session_id=session_id)
        except SQLAlchemyError as e:
            raise FloatChatException("Failed to clear session data", e)

    def execute_sql(self, query: str, params: dict = None):
        """Execute arbitrary SQL and return rows as list of dicts."""
        try:
            with self.engine.begin() as conn:
                result = conn.execute(text(query), params or {})
                rows = result.fetchall()

            output = [dict(r._mapping) for r in rows]
            self.logger.info("✅ SQL executed", query=query, rows=len(output))
            return output
        except SQLAlchemyError as e:
            raise FloatChatException("Failed to execute SQL query", e)
