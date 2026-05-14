import time
import mlflow
import logging
import os
import shutil
import chromadb
import boto3
import aiofiles

from typing import Annotated
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.llms.ollama import Ollama

from get_embedding_function import get_embedding_function
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi.middleware.cors import CORSMiddleware


# ================= LOGGING =================

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ================= CONFIGURATION =================

def get_env_variable(name: str, default: str = None):
    value = os.getenv(name, default)
    if value is None:
        raise ValueError(f"Variable d'environnement manquante: {name}")
    return value


CHROMA_HOST = get_env_variable("CHROMA_HOST", "chromadb.chromadb.svc.cluster.local")
CHROMA_PORT = int(get_env_variable("CHROMA_PORT", "8000"))
CHROMA_COLLECTION = get_env_variable("CHROMA_COLLECTION", "pdf_documents")
DATA_PATH = get_env_variable("DATA_PATH", "/app/data")

OLLAMA_MODEL = get_env_variable("OLLAMA_MODEL", "mistral")
OLLAMA_BASE_URL = get_env_variable(
    "OLLAMA_BASE_URL",
    "http://ollama.ollama.svc.cluster.local:11434"
)

MLFLOW_TRACKING_URI = get_env_variable(
    "MLFLOW_TRACKING_URI",
    "http://mlflow.mlflow.svc.cluster.local:5000"
)

ENABLE_MLFLOW_ASK = os.getenv("ENABLE_MLFLOW_ASK", "false").lower() == "true"

MINIO_ENDPOINT = get_env_variable(
    "MINIO_ENDPOINT",
    "http://minio.minio.svc.cluster.local:9000"
)
MINIO_ACCESS_KEY = get_env_variable("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = get_env_variable("MINIO_SECRET_KEY", "password123")
MINIO_BUCKET = get_env_variable("MINIO_BUCKET", "pdf-documents")
MINIO_EXPECTED_BUCKET_OWNER = get_env_variable("MINIO_EXPECTED_BUCKET_OWNER", "minio")


mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("RAG-PDF-Demo")


# ================= FASTAPI =================

app = FastAPI(title="RAG PDF API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://rag-ui.local",
        "http://fastapi.local"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.on_event("startup")
def startup_event():
    logger.info("Application FastAPI démarrée")
    logger.info(f"CHROMA_HOST={CHROMA_HOST}")
    logger.info(f"CHROMA_PORT={CHROMA_PORT}")
    logger.info(f"CHROMA_COLLECTION={CHROMA_COLLECTION}")
    logger.info(f"DATA_PATH={DATA_PATH}")
    logger.info(f"OLLAMA_BASE_URL={OLLAMA_BASE_URL}")
    logger.info(f"OLLAMA_MODEL={OLLAMA_MODEL}")
    logger.info(f"MLFLOW_TRACKING_URI={MLFLOW_TRACKING_URI}")
    logger.info(f"ENABLE_MLFLOW_ASK={ENABLE_MLFLOW_ASK}")


# ================= PROMPT =================

PROMPT_TEMPLATE = """
Tu es un assistant RAG spécialisé dans l'analyse de documents PDF.

Réponds uniquement à partir du contexte fourni.

Règles :
- N'invente aucune information.
- Si l'information n'est pas présente dans le contexte, réponds exactement :
  "Je ne trouve pas cette information dans le document."
- Réponds de manière concise.
- Donne une réponse en 3 à 5 lignes maximum.
- Ne répète pas le contexte.
- Va directement à l'essentiel.

Contexte :
{context}

---

Question :
{question}

---

Réponse concise :
"""

# ================= UTILITAIRES =================

def clear_database() -> None:
    client = chromadb.HttpClient(
        host=CHROMA_HOST,
        port=CHROMA_PORT
    )

    try:
        client.delete_collection(CHROMA_COLLECTION)
        logger.info("Collection Chroma supprimée")
    except Exception:
        logger.info("Collection Chroma inexistante ou déjà supprimée")


def clear_data_folder() -> None:
    data_dir = Path(DATA_PATH)
    data_dir.mkdir(parents=True, exist_ok=True)

    for item in data_dir.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)

    logger.info("Dossier data vidé")


def load_documents():
    loader = PyPDFDirectoryLoader(DATA_PATH)
    return loader.load()


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_documents(documents)


def get_vector_db():
    client = chromadb.HttpClient(
        host=CHROMA_HOST,
        port=CHROMA_PORT
    )

    return Chroma(
        client=client,
        collection_name=CHROMA_COLLECTION,
        embedding_function=get_embedding_function()
    )


def get_llm():
    return Ollama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        timeout=120,
        num_predict=120,
        temperature=0,
    )


def upload_to_minio(file_path: str, filename: str) -> None:
    s3 = boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY
    )

    try:
      s3.head_bucket(
         Bucket=MINIO_BUCKET,
         ExpectedBucketOwner=MINIO_EXPECTED_BUCKET_OWNER
      )
    except Exception:
        try:
            s3.create_bucket(Bucket=MINIO_BUCKET)
            logger.info(f"Bucket MinIO créé: {MINIO_BUCKET}")
        except Exception as e:
            logger.warning(f"Impossible de créer/vérifier le bucket MinIO: {str(e)}")

    s3.upload_file(
        file_path,
        MINIO_BUCKET,
        filename,
        ExtraArgs={
            "ExpectedBucketOwner": MINIO_EXPECTED_BUCKET_OWNER
        }
    )

    logger.info(f"PDF envoyé vers MinIO: {filename}")


# ================= INGESTION =================

def ingest_documents():
    documents = load_documents()
    logger.info(f"Nombre de documents chargés: {len(documents)}")

    if not documents:
        raise ValueError("Aucun PDF lisible n'a été trouvé dans le dossier data.")

    chunks = split_documents(documents)
    logger.info(f"Nombre de chunks créés: {len(chunks)}")

    if not chunks:
        raise ValueError("Aucun chunk n'a été créé à partir du PDF.")

    db = get_vector_db()
    db.add_documents(chunks)

    logger.info("Documents ajoutés dans Chroma")

    return {
        "documents_loaded": len(documents),
        "chunks_created": len(chunks),
    }


# ================= RAG =================

def query_rag(question: str):
    total_start = time.time()

    db = get_vector_db()

    retrieval_start = time.time()
    filtered_docs = db.similarity_search(question, k=5)
    retrieval_time = time.time() - retrieval_start
    
    logger.info(f"Temps retrieval Chroma: {retrieval_time:.2f}s")
    
    if not filtered_docs:
        return {
            "question": question,
            "answer": "Aucun contenu pertinent n'a été trouvé dans le PDF indexé.",
            "sources": []
        }

    unique_docs = []
    seen = set()

    for doc in filtered_docs:
        content = doc.page_content.strip()

        if content and content not in seen and len(content) > 50:
            unique_docs.append(doc)
            seen.add(content)

    if not unique_docs:
        return {
            "question": question,
            "answer": "Aucun contenu pertinent n'a été trouvé dans le PDF indexé.",
            "sources": []
        }

    selected_docs = unique_docs[:4]

    context = "\n\n---\n\n".join(
        [doc.page_content[:500] for doc in selected_docs]
    )

    logger.info(f"Nombre de chunks utilisés: {len(selected_docs)}")
    logger.info(f"Taille du contexte envoyé au modèle: {len(context)} caractères")

    prompt = PROMPT_TEMPLATE.format(
        context=context,
        question=question
    )

    llm = get_llm()

    llm_start = time.time()

    try:
        answer = llm.invoke(prompt)

        if not answer or len(str(answer).strip()) == 0:
            answer = "Je ne trouve pas cette information dans le document."

    except Exception as e:
        logger.error(f"Erreur Ollama: {str(e)}")
        answer = "Erreur lors de la génération de la réponse."

    llm_time = time.time() - llm_start
    total_time = time.time() - total_start

    logger.info(f"Temps génération Ollama: {llm_time:.2f}s")
    logger.info(f"Temps total query_rag: {total_time:.2f}s")

    sources = []
    for doc in selected_docs:
        source = doc.metadata.get("source", "unknown")
        if source not in sources:
            sources.append(source)

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "retrieval_time": round(retrieval_time, 2),
        "llm_time": round(llm_time, 2),
        "total_time": round(total_time, 2),
    }


# ================= ROUTES =================

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": OLLAMA_MODEL,
        "chroma_host": CHROMA_HOST,
        "chroma_collection": CHROMA_COLLECTION,
    }


@app.post("/upload")
async def upload_pdf(
    file: Annotated[UploadFile, File(...)],
    reset_db: Annotated[bool, Form(False)]
):
    start_time = time.time()
    logger.info(f"Upload reçu: {file.filename}")

    try:
        if not file.filename:
            return {
                "success": False,
                "error": "Aucun fichier fourni."
            }

        if not file.filename.lower().endswith(".pdf"):
            return {
                "success": False,
                "error": "Seuls les fichiers PDF sont autorisés."
            }

        Path(DATA_PATH).mkdir(parents=True, exist_ok=True)

        if reset_db:
            clear_database()
            clear_data_folder()
        else:
            logger.warning("Réindexation sans reset: risque de duplication")

        file_path = os.path.join(DATA_PATH, file.filename)

        async with aiofiles.open(file_path, "wb") as buffer:
            content = await file.read()
            await buffer.write(content)

        upload_to_minio(file_path, file.filename)

        result = ingest_documents()
        indexation_time = time.time() - start_time

        with mlflow.start_run(run_name="upload-indexation"):
            mlflow.log_param("filename", file.filename)
            mlflow.log_param("model", OLLAMA_MODEL)
            mlflow.log_param("embedding_model", "HuggingFaceEmbeddings")
            mlflow.log_param("chunk_size", 800)
            mlflow.log_param("chunk_overlap", 80)
            mlflow.log_metric("documents_loaded", result["documents_loaded"])
            mlflow.log_metric("nombre_chunks", result["chunks_created"])
            mlflow.log_metric("temps_indexation", indexation_time)

        return {
            "success": True,
            "message": "PDF uploadé et indexé avec succès.",
            "filename": file.filename,
            "result": result,
            "indexation_time": round(indexation_time, 2)
        }

    except Exception as e:
        logger.exception("Erreur pendant upload/indexation")
        return {
            "success": False,
            "error": str(e)
        }


@app.post("/index-from-existing-pdf")
def index_from_existing_pdf():
    start_time = time.time()
    logger.info("Indexation lancée depuis Kubeflow")

    try:
        result = ingest_documents()
        indexation_time = time.time() - start_time

        with mlflow.start_run(run_name="kubeflow-indexation"):
            mlflow.log_param("source", "Kubeflow")
            mlflow.log_param("model", OLLAMA_MODEL)
            mlflow.log_param("embedding_model", "HuggingFaceEmbeddings")
            mlflow.log_param("chunk_size", 800)
            mlflow.log_param("chunk_overlap", 80)
            mlflow.log_metric("documents_loaded", result["documents_loaded"])
            mlflow.log_metric("nombre_chunks", result["chunks_created"])
            mlflow.log_metric("temps_indexation", indexation_time)

        return {
            "success": True,
            "message": "Indexation lancée depuis Kubeflow",
            "result": result,
            "indexation_time": round(indexation_time, 2)
        }

    except Exception as e:
        logger.exception("Erreur pendant indexation Kubeflow")
        return {
            "success": False,
            "error": str(e)
        }


@app.post("/ask")
def ask_question(
    question: Annotated[str, Form(...)]
):
    start_time = time.time()
    logger.info(f"Question reçue: {question}")

    try:
        if not question.strip():
            return {
                "success": False,
                "error": "La question ne peut pas être vide."
            }

        result = query_rag(question)
        response_time = time.time() - start_time

        if ENABLE_MLFLOW_ASK:
            with mlflow.start_run(run_name="question-answering"):
                mlflow.log_param("model", OLLAMA_MODEL)
                mlflow.log_param("question", question[:250])
                mlflow.log_metric("temps_reponse", response_time)
                mlflow.log_metric("nombre_sources", len(result.get("sources", [])))
                mlflow.log_metric("retrieval_time", result.get("retrieval_time", 0))
                mlflow.log_metric("llm_time", result.get("llm_time", 0))

        return {
            "success": True,
            **result,
            "response_time": round(response_time, 2)
        }

    except Exception as e:
        logger.exception("Erreur pendant question/réponse")
        return {
            "success": False,
            "error": str(e)
        }