# server.py
import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from utils.netcdf_loader import NetCDFLoader
from utils.db_connector import DBLoader
from src.float_ingestion import FloatIngestor
from src.chat_orchestrator import ChatOrchestrator
from src.float_rag import FloatChatRAG
from exception.custom_exception import FloatChatException
from logger.custom_logger import CustomLogger
from utils.nl_router import parse_nl_question   # 👈 NEW

app = FastAPI(title="FloatChat Backend API")
logger = CustomLogger().get_logger(__file__)

# --------------------------------
# CORS
# --------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------
# Global error handler
# --------------------------------
@app.exception_handler(FloatChatException)
async def floatchat_exception_handler(request: Request, exc: FloatChatException):
    logger.error("FloatChatException", path=str(request.url), detail=str(exc))
    return JSONResponse(status_code=500, content={"status": "error", "detail": str(exc)})


# -------------------------------
# 1) Session management
# -------------------------------
@app.post("/start-session")
def start_session():
    try:
        db = DBLoader()
        session_id = db.new_session()
        return {"status": "success", "session_id": session_id}
    except Exception as e:
        logger.error("Failed to start session", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to start session: {e}")


@app.delete("/end-session/{session_id}")
def end_session(session_id: str):
    try:
        db = DBLoader()
        db.clear_session(session_id)

        faiss_dir = os.path.join("faiss_index", session_id)
        if os.path.exists(faiss_dir):
            shutil.rmtree(faiss_dir)

        return {"status": "success", "message": f"Session {session_id} cleared"}
    except Exception as e:
        logger.error("Failed to end session", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to end session: {e}")


# -------------------------------
# 2) File upload → parse → store
# -------------------------------
@app.post("/upload-nc/{session_id}")
async def upload_nc(session_id: str, file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(".nc"):
            raise HTTPException(status_code=400, detail="Only .nc files supported")

        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        loader = NetCDFLoader()
        df = loader.load_netcdf(file_path)
        if df.empty:
            raise HTTPException(status_code=400, detail="Parsed DataFrame is empty")

        db = DBLoader()
        db.create_table()
        db.insert_profiles(df, session_id=session_id)

        ingestor = FloatIngestor(session_id=session_id)
        ingestor.build_retriever(df)

        preview = df.head(10).to_dict(orient="records")
        return {"status": "success", "rows": len(df), "preview": preview}

    except FloatChatException as fe:
        logger.error("Upload pipeline failed (FloatChatException)", session_id=session_id, detail=str(fe))
        raise HTTPException(status_code=500, detail=str(fe))
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unexpected upload error", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


# -------------------------------
# 3) Fetch profiles
# -------------------------------
@app.get("/profiles/{session_id}")
def get_profiles(session_id: str, limit: int = 100):
    try:
        db = DBLoader()
        rows = db.query_profiles(session_id, limit)
        return {"status": "success", "data": rows}
    except Exception as e:
        logger.error("Failed to fetch profiles", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to fetch profiles: {e}")


# -------------------------------
# 4) Conversational Chat
# -------------------------------
@app.post("/chat/{session_id}")
async def chat_with_data(session_id: str, payload: dict, request: Request):
    try:
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Invalid payload; expected JSON object")

        query = payload.get("query")
        history = payload.get("history", [])
        stream = bool(payload.get("stream", False))

        if not query or not isinstance(query, str):
            raise HTTPException(status_code=400, detail="Query text is required")

        # 👇 NEW: Parse query to detect if rows should be returned
        parsed = parse_nl_question(query)
        include_rows = parsed.get("include_rows", False)

        db_url = os.getenv("DATABASE_URL", "postgresql://floatchat:yourpassword@localhost:5432/floatchat")
        faiss_dir = "faiss_index"
        orchestrator = ChatOrchestrator(session_id, db_url, faiss_dir)

        if stream:
            rag = FloatChatRAG(session_id)
            rag.load_retriever_from_faiss(os.path.join(faiss_dir, session_id))

            def token_generator():
                try:
                    for chunk in rag.stream(query, chat_history=history):
                        yield chunk
                except Exception as e:
                    logger.error("Streaming error", session_id=session_id, error=str(e))
                    yield f"\n[ERROR] {e}"

            return StreamingResponse(token_generator(), media_type="text/plain")

        # Non-streaming
        result = orchestrator.chat(query, include_rows=include_rows)   # 👈 pass flag
        return {
            "status": "success",
            "answer": result.get("answer"),
            "dataframe": result.get("dataframe"),
            "sources": result.get("sources"),
        }

    except HTTPException:
        raise
    except FloatChatException as fe:
        logger.error("Chat failed (FloatChatException)", session_id=session_id, detail=str(fe))
        raise HTTPException(status_code=500, detail=f"Chat failed: {fe}")
    except Exception as e:
        logger.error("Chat endpoint failed", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")


# -------------------------------
# 5) Root
# -------------------------------
@app.get("/")
def root():
    return {"message": "🌊 FloatChat Backend API is running 🚀"}
