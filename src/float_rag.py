# src/float_rag.py
import sys
import os
import json
import re
from operator import itemgetter
from typing import List, Optional, Dict, Any, Generator

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_community.vectorstores import FAISS

from utils.model_loader import ModelLoader
from exception.custom_exception import FloatChatException
from logger.custom_logger import CustomLogger

from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# Optional Google exceptions (present only when google packages are installed)
try:
    from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable
except Exception:  # pragma: no cover
    ResourceExhausted = ServiceUnavailable = tuple()  # type: ignore


# ---------- helpers ----------

def _coerce_to_text(x) -> str:
    """Ensure any object becomes a plain string (safe for embeddings/retriever)."""
    if isinstance(x, str):
        return x
    if hasattr(x, "content"):
        try:
            return str(getattr(x, "content"))
        except Exception:
            pass
    try:
        return json.dumps(x, default=str)
    except Exception:
        return str(x)


# ---------- RAG class ----------

class FloatChatRAG:
    """
    Conversational RAG pipeline for ARGO float data with provider failover (Google <-> Groq).

    New in this version:
    - `include_rows` flag in `invoke()`/`stream()` to control whether the model should
      surface coordinates/rows if they appear in context. When False (default), the
      prompt discourages listing specific positions unless explicitly asked.
    """

    def __init__(self, session_id: Optional[str], retriever=None):
        try:
            self.logger = CustomLogger().get_logger(__file__)
            self.session_id = session_id

            # Load LLM (with provider fallback at load time)
            self.llm = self._load_llm()
            if isinstance(self.llm, str):
                raise FloatChatException(
                    f"Invalid LLM instance: got string '{self.llm}'."
                )

            # Prompts
            self.contextualize_prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(
                    "You are an assistant for exploring ARGO float data. "
                    "Rewrite the user question to make it more explicit if needed."
                ),
                HumanMessagePromptTemplate.from_template("{input}\nChat history: {chat_history}")
            ])

            # NOTE: We pass include_rows into the final QA prompt to control output style.
            self.qa_prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(
                    "You are a helpful oceanography assistant. "
                    "Answer questions using the retrieved ARGO float data context.\n"
                    "- If you are unsure, say you don’t know. Do not hallucinate.\n"
                    "- include_rows={include_rows}.\n"
                    "  • When include_rows is false: summarize the answer without listing specific coordinates, "
                    "    float IDs, or per-row details unless the user explicitly asked for them.\n"
                    "  • When include_rows is true: if the context contains coordinates/rows (lat/lon/time/ids), "
                    "    extract and present them succinctly (e.g., a short bullet list or compact table). "
                    "    Never invent rows; only use what’s in context."
                ),
                HumanMessagePromptTemplate.from_template(
                    "Context:\n{context}\n\nQuestion: {input}"
                ),
            ])

            # Lazy init
            self.retriever = retriever
            self.chain = None
            if self.retriever:
                self._build_chain()

            self.logger.info("FloatChatRAG initialized", session_id=self.session_id)

        except Exception as e:
            raise FloatChatException("Initialization error in FloatChatRAG", e)

    # ---------- Public API ----------

    def load_retriever_from_faiss(
        self,
        index_path: str,
        k: int = 5,
        index_name: str = "index",
        search_type: str = "similarity",
        search_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Load FAISS retriever from a session-specific FAISS index."""
        try:
            if not os.path.isdir(index_path):
                raise FileNotFoundError(f"FAISS index not found: {index_path}")

            embeddings = ModelLoader().load_embeddings()
            vectorstore = FAISS.load_local(
                index_path,
                embeddings,
                index_name=index_name,
                allow_dangerous_deserialization=True,
            )

            if search_kwargs is None:
                search_kwargs = {"k": k}

            self.retriever = vectorstore.as_retriever(
                search_type=search_type, search_kwargs=search_kwargs
            )
            self._build_chain()

            self.logger.info("FAISS retriever loaded", path=index_path, session_id=self.session_id)
            return self.retriever

        except Exception as e:
            raise FloatChatException("Failed to load retriever from FAISS", e)

    def invoke(
        self,
        user_input: str,
        chat_history: Optional[List[BaseMessage]] = None,
        *,
        include_rows: bool = False,  # 👈 NEW
    ) -> str:
        """
        Run the Conversational RAG pipeline and return the full answer.
        Includes runtime failover on LLM rate-limit/quota/service-unavailable and
        model config errors (e.g., model_not_found, decommissioned).

        `include_rows` guides the QA prompt to either summarize (False) or surface
        per-row details if present in context (True).
        """
        try:
            if not self.chain:
                raise FloatChatException("RAG chain not initialized. Load retriever first.")

            # Force input to string
            if not isinstance(user_input, str):
                user_input = _coerce_to_text(user_input)

            chat_history = chat_history or []
            payload = {
                "input": user_input,
                "chat_history": chat_history,
                "include_rows": include_rows,
            }

            try:
                answer = self.chain.invoke(payload)
            except Exception as e:
                if self._should_failover(e):
                    self.logger.warning(
                        "Primary LLM error; failing over to alternate provider",
                        provider=self._current_provider(),
                        error=str(e),
                    )
                    self._failover_llm_and_rebuild()
                    answer = self.chain.invoke(payload)  # one retry after failover
                else:
                    raise

            if not answer:
                return "No answer generated."

            self.logger.info(
                "RAG invoked",
                session_id=self.session_id,
                user_input=user_input,
                provider=self._current_provider(),
                include_rows=include_rows,
            )
            return answer

        except Exception as e:
            raise FloatChatException("Error invoking FloatChatRAG", e)

    def stream(
        self,
        user_input: str,
        chat_history: Optional[List[BaseMessage]] = None,
        *,
        include_rows: bool = False,  # 👈 NEW
    ) -> Generator[str, None, None]:
        """
        Stream the answer token-by-token. If the active LLM errors on limits or model config,
        fail over to the alternate provider and restart the stream once.

        `include_rows` is passed to the prompt to shape output.
        """
        try:
            if not self.chain:
                raise FloatChatException("RAG chain not initialized. Load retriever first.")

            if not isinstance(user_input, str):
                user_input = _coerce_to_text(user_input)

            chat_history = chat_history or []
            payload = {
                "input": user_input,
                "chat_history": chat_history,
                "include_rows": include_rows,
            }

            yielded_any = False
            try:
                for chunk in self.chain.stream(payload):
                    yielded_any = True
                    if chunk:
                        yield chunk
            except Exception as e:
                # Only retry if we haven't streamed anything yet (to avoid duplicate output)
                if not yielded_any and self._should_failover(e):
                    self.logger.warning(
                        "Streaming failed due to provider issue; failing over and retrying stream",
                        provider=self._current_provider(),
                        error=str(e),
                    )
                    self._failover_llm_and_rebuild()
                    for chunk in self.chain.stream(payload):
                        if chunk:
                            yield chunk
                else:
                    raise

            self.logger.info(
                "RAG streamed",
                session_id=self.session_id,
                user_input=user_input,
                provider=self._current_provider(),
                include_rows=include_rows,
            )

        except Exception as e:
            raise FloatChatException("Error streaming FloatChatRAG", e)

    # ---------- Internals ----------

    def _load_llm(self):
        try:
            # Uses fallback across providers based on env/config order
            llm = ModelLoader().load_llm_fallback()
            if not llm:
                raise ValueError("LLM could not be loaded")
            return llm
        except Exception as e:
            raise FloatChatException("Error loading LLM in FloatChatRAG", e)

    @staticmethod
    def _format_docs(docs) -> str:
        """Format retrieved documents into a string context."""
        return "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)

    def _retrieve(self, q) -> List[Any]:
        """Wrapper that guarantees a string before hitting the retriever."""
        if not self.retriever:
            raise FloatChatException("Retriever not initialized before _retrieve()")
        safe_q = _coerce_to_text(q)
        return self.retriever.invoke(safe_q)

    def _build_chain(self):
        """Build the LCEL graph: contextualize → retrieve → answer."""
        try:
            if not self.retriever:
                raise FloatChatException("Retriever not set before building chain")

            if isinstance(self.llm, str):
                raise FloatChatException(
                    f"Cannot build chain: LLM is a string ('{self.llm}') instead of a Runnable."
                )

            # 1) Rewrite question with context → as plain string
            question_rewriter = (
                {"input": itemgetter("input"), "chat_history": itemgetter("chat_history")}
                | self.contextualize_prompt
                | self.llm
                | StrOutputParser()                # ensures a string
                | RunnableLambda(_coerce_to_text)  # extra guard
            )

            # 2) Retrieve docs using our safe wrapper (we control input type)
            retrieve_docs = (
                question_rewriter
                | RunnableLambda(self._retrieve)   # always string to retriever
                | RunnableLambda(self._format_docs)
            )

            # 3) Final QA step (answer with retrieved context)
            #    We thread include_rows from the payload into the prompt.
            self.chain = (
                {
                    "context": retrieve_docs,
                    "input": itemgetter("input"),
                    "chat_history": itemgetter("chat_history"),
                    "include_rows": itemgetter("include_rows"),
                }
                | self.qa_prompt
                | self.llm
                | StrOutputParser()
            )

        except Exception as e:
            raise FloatChatException("Error building FloatChatRAG chain", e)

    # ---------- Failover utilities ----------

    def _is_rate_limit_error(self, err: Exception) -> bool:
        """Detect errors that should trigger provider failover (limits/outages)."""
        msg = str(err).lower()
        if isinstance(err, (ResourceExhausted, ServiceUnavailable)):
            return True
        patterns = [
            r"\b429\b",
            r"rate[-\s]?limit",
            r"\bquota\b",
            r"resource exhausted",
            r"service unavailable",
            r"temporar(il|y) unavailable",
            r"timeout",
        ]
        return any(re.search(p, msg) for p in patterns)

    def _is_model_config_error(self, err: Exception) -> bool:
        """Detect provider/model configuration problems that should also trigger failover."""
        msg = str(err).lower()
        patterns = [
            r"model[_\s-]?not[_\s-]?found",
            r"\bmodel\b.*\bdoes not exist\b",
            r"decommissioned",
            r"no access",
            r"not authorized",
            r"forbidden",
            r"\b403\b",
        ]
        return any(re.search(p, msg) for p in patterns)

    def _should_failover(self, err: Exception) -> bool:
        """Union of rate-limit and model-config errors."""
        return self._is_rate_limit_error(err) or self._is_model_config_error(err)

    def _current_provider(self) -> str:
        """Best-effort identification of the active provider."""
        cls = self.llm.__class__.__name__.lower()
        if "google" in cls:
            return "google"
        if "groq" in cls:
            return "groq"
        return "unknown"

    def _failover_llm_and_rebuild(self):
        """Switch to the alternate provider and rebuild the chain."""
        try:
            cur = self._current_provider()
            order = ["groq", "google"] if cur == "google" else ["google", "groq"]
            self.llm = ModelLoader().load_llm_fallback(order=order)
            self._build_chain()
            self.logger.info("LLM failover complete", from_provider=cur, to_provider=self._current_provider())
        except Exception as e:
            raise FloatChatException("Provider failover failed", e)
