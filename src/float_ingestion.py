import os
import json
import uuid
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from utils.model_loader import ModelLoader
from exception.custom_exception import FloatChatException
from logger.custom_logger import CustomLogger


class FaissManager:
    """
    Handles FAISS vector DB storage & metadata for session-specific indexes.
    """
    def __init__(self, index_dir: Path, model_loader: Optional[ModelLoader] = None):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.meta_path = self.index_dir / "ingested_meta.json"
        self._meta: Dict[str, Any] = {"rows": {}}

        if self.meta_path.exists():
            try:
                self._meta = json.loads(self.meta_path.read_text(encoding="utf-8")) or {"rows": {}}
            except Exception:
                self._meta = {"rows": {}}

        self.logger = CustomLogger().get_logger(__file__)
        self.model_loader = model_loader or ModelLoader()
        self.emb = self.model_loader.load_embeddings()
        self.vs: Optional[FAISS] = None

    def _exists(self) -> bool:
        return (self.index_dir / "index.faiss").exists() and (self.index_dir / "index.pkl").exists()

    def _fingerprint(self, row: Dict[str, Any]) -> str:
        """
        Unique key for a row (float_id + cycle_number + depth).
        """
        src = str(row.get("float_id", "unknown"))
        cycle = str(row.get("cycle_number", "NA"))
        depth = str(row.get("depth", "NA"))
        return f"{src}::{cycle}::{depth}"

    def _save_meta(self):
        self.meta_path.write_text(json.dumps(self._meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_rows(self, docs: List[Document]):
        if self.vs is None:
            raise RuntimeError("Call load_or_create() before add_rows().")

        new_docs: List[Document] = []
        for d in docs:
            key = self._fingerprint(d.metadata or {})
            if key in self._meta["rows"]:
                continue
            self._meta["rows"][key] = True
            new_docs.append(d)

        if new_docs:
            self.vs.add_documents(new_docs)
            self.vs.save_local(str(self.index_dir))
            self._save_meta()

        return len(new_docs)

    def load_or_create(self, texts: Optional[List[str]] = None, metadatas: Optional[List[dict]] = None):
        if self._exists():
            self.vs = FAISS.load_local(
                str(self.index_dir),
                embeddings=self.emb,
                allow_dangerous_deserialization=True,
            )
            return self.vs

        if not texts:
            raise FloatChatException("No existing FAISS index and no data to create one")

        self.vs = FAISS.from_texts(texts=texts, embedding=self.emb, metadatas=metadatas or [])
        self.vs.save_local(str(self.index_dir))
        return self.vs


class FloatIngestor:
    """
    Converts DataFrame rows → embeddings → FAISS index for semantic search.
    """

    def __init__(self, faiss_base: str = "faiss_index", session_id: Optional[str] = None):
        self.model_loader = ModelLoader()
        self.session_id = session_id or str(uuid.uuid4())

        self.faiss_base = Path(faiss_base)
        self.faiss_base.mkdir(parents=True, exist_ok=True)

        self.faiss_dir = self.faiss_base / self.session_id
        self.faiss_dir.mkdir(parents=True, exist_ok=True)

        self.logger = CustomLogger().get_logger(__file__)
        self.logger.info("FloatIngestor initialized", session_id=self.session_id, faiss_dir=str(self.faiss_dir))

    def _summarize_row(self, row: pd.Series) -> str:
        return (
            f"Float {row.get('float_id', 'unknown')}, "
            f"Cycle {row.get('cycle_number', 'NA')}, "
            f"Temperature {row.get('temperature', 'NA')} °C, "
            f"Salinity {row.get('salinity', 'NA')} PSU, "
            f"Depth {row.get('depth', 'NA')} m, "
            f"Lat {row.get('lat', 'NA')}, Lon {row.get('lon', 'NA')}, "
            f"Time {row.get('time', 'NA')}."
        )

    def build_retriever(self, df: pd.DataFrame, k: int = 5):
        """
        Ingest DataFrame into FAISS and return retriever.
        """
        try:
            fm = FaissManager(self.faiss_dir, self.model_loader)

            texts = df.apply(self._summarize_row, axis=1).tolist()
            metas = df.to_dict(orient="records")

            vs = fm.load_or_create(texts=texts, metadatas=metas)
            added = fm.add_rows([Document(page_content=texts[i], metadata=metas[i]) for i in range(len(texts))])

            self.logger.info("FAISS index updated", added=added, index=str(self.faiss_dir))

            return vs.as_retriever(search_type="similarity", search_kwargs={"k": k})

        except Exception as e:
            self.logger.error("Failed to build retriever", error=str(e))
            raise FloatChatException("Failed to build retriever", e)
