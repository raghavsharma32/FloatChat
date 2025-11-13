# utils/model_loader.py
import os
import re
from dotenv import load_dotenv
from typing import Optional, List

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from utils.config_loader import load_config
from logger.custom_logger import CustomLogger
from exception.custom_exception import FloatChatException

# Optional Google API errors for retry detection
try:
    from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable
except Exception:  # pragma: no cover
    ResourceExhausted = ServiceUnavailable = tuple()


class ModelLoader:
    """
    Loads embeddings and LLMs, with provider fallback:
    - Preferred order is configurable; default ['google', 'groq'].
    - Falls back on common transient/429 errors.
    """

    def __init__(self):
        load_dotenv()
        self.logger = CustomLogger().get_logger(__file__)
        self.config = load_config()

        self.google_key = os.getenv("GOOGLE_API_KEY")
        self.groq_key = os.getenv("GROQ_API_KEY")

        if not (self.google_key or self.groq_key):
            raise FloatChatException("❌ No API keys found (GOOGLE_API_KEY / GROQ_API_KEY).")

        self.logger.info("✅ ModelLoader initialized", config_keys=list(self.config.keys()))

    # ---------- Embeddings ----------

    def load_embeddings(self):
        """Load embeddings (Google preferred, with HuggingFace fallback)."""
        try:
            emb_cfg = self.config.get("embedding_model", {})
            provider = (emb_cfg.get("provider") or "google").lower()
            model_name = emb_cfg.get("model_name") or "models/text-embedding-004"

            # Auto-upgrade legacy names
            if model_name in {"embedding-001", "models/embedding-001"}:
                self.logger.warning(
                    "Embedding model '%s' deprecated; switching to 'models/text-embedding-004'",
                    model_name,
                )
                model_name = "models/text-embedding-004"

            if provider == "google":
                if not self.google_key:
                    raise FloatChatException("❌ GOOGLE_API_KEY not set for Google embeddings.")
                self.logger.info("📥 Loading Google embeddings", model=model_name)
                return GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=self.google_key)

            elif provider == "huggingface":
                from langchain_community.embeddings import HuggingFaceEmbeddings
                name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
                self.logger.info("📥 Loading HuggingFace embeddings", model=name)
                return HuggingFaceEmbeddings(model_name=name)

            else:
                raise FloatChatException(f"Unsupported embeddings provider: {provider}")

        except Exception as e:
            raise FloatChatException("Failed to load embedding model", e)

    # ---------- LLMs with fallback ----------

    def _make_google_chat(self, cfg: dict):
        """Instantiate Google Gemini chat model."""
        model = cfg.get("model_name", "gemini-1.5-flash")
        temperature = float(cfg.get("temperature", 0.2))
        max_tokens = int(cfg.get("max_output_tokens", 2048))

        if not self.google_key:
            raise FloatChatException("❌ GOOGLE_API_KEY not set for Google LLM.")
        self.logger.info("📥 Loading Google LLM", model=model)

        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=self.google_key,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    def _make_groq_chat(self, cfg: dict):
        """Instantiate Groq LLM."""
        model = cfg.get("model_name", "llama-3.3-70b-versatile")
        temperature = float(cfg.get("temperature", 0.2))

        if not self.groq_key:
            raise FloatChatException("❌ GROQ_API_KEY not set for Groq LLM.")
        self.logger.info("📥 Loading Groq LLM", model=model)

        return ChatGroq(
            model=model,
            api_key=self.groq_key,
            temperature=temperature,
        )

    @staticmethod
    def _is_retryable_llm_error(err: Exception) -> bool:
        """Check if the error is transient/retryable (quota, 429, etc.)."""
        msg = str(err).lower()

        if isinstance(err, (ResourceExhausted, ServiceUnavailable)):
            return True

        patterns = [
            r"rate[-\s]?limit", r"quota", r"\b429\b",
            r"resource exhausted", r"service unavailable",
            r"timeout", r"temporar(il|y) unavailable",
        ]
        return any(re.search(p, msg) for p in patterns)

    def load_llm_fallback(self, order: Optional[List[str]] = None):
        """
        Load LLM with provider fallback.
        Default order: env LLM_PROVIDER if set, else ['google', 'groq'].
        """
        try:
            llm_block = self.config.get("llm", {})

            # Env override
            env_pref = os.getenv("LLM_PROVIDER", "").strip().lower()
            if order is None:
                if env_pref in ("google", "groq"):
                    order = [env_pref] + [p for p in ["google", "groq"] if p != env_pref]
                else:
                    order = ["google", "groq"]

            errors = []
            for provider in order:
                cfg = llm_block.get(provider)
                if not cfg:
                    errors.append(f"⚠️ Config block missing for provider '{provider}'.")
                    continue

                try:
                    if provider == "google":
                        return self._make_google_chat(cfg)
                    elif provider == "groq":
                        return self._make_groq_chat(cfg)
                    else:
                        errors.append(f"Unsupported provider '{provider}'.")
                except Exception as e:
                    if self._is_retryable_llm_error(e):
                        self.logger.warning(
                            "⚠️ LLM provider %s failed; trying next provider...",
                            provider,
                            error=str(e),
                        )
                        errors.append(f"{provider} failed: {e}")
                        continue
                    # Non-retryable → stop immediately
                    raise

            # If exhausted all providers
            raise FloatChatException("All LLM providers failed: " + " | ".join(errors))

        except Exception as e:
            raise FloatChatException("Failed to load LLM with fallback", e)

    def load_llm(self):
        """Backwards-compatible wrapper → always use fallback loader."""
        return self.load_llm_fallback()
