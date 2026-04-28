import mlflow
import time
import os
import shutil
import chromadb
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_community.llms.ollama import Ollama

from get_embedding_function import get_embedding_function


# ================= CONFIGURATION =================

def get_env_variable(name: str, default: str = None):
    value = os.getenv(name, default)
    if value is None:
        raise ValueError(f"❌ Variable d'environnement manquante: {name}")
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

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("RAG-PDF-Demo")

app = FastAPI(title="RAG PDF API")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


PROMPT_TEMPLATE = """
Réponds à la question uniquement à partir du contexte suivant :

{context}

---

Question : {question}
Réponse :
"""


def clear_database() -> None:
    client = chromadb.HttpClient(
        host=CHROMA_HOST,
        port=CHROMA_PORT
    )

    try:
        client.delete_collection(CHROMA_COLLECTION)
        print("✅ Collection Chroma supprimée")
    except Exception:
        print("ℹ️ Collection Chroma inexistante ou déjà supprimée")

def clear_data_folder() -> None:
    """
    Vide le dossier data sans supprimer le dossier lui-même.
    """
    data_dir = Path(DATA_PATH)
    data_dir.mkdir(parents=True, exist_ok=True)

    for item in data_dir.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


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


def ingest_documents():
    documents = load_documents()
    print("DOCS:", len(documents))

    if not documents:
        raise ValueError("Aucun PDF lisible n'a été trouvé dans le dossier data.")

    chunks = split_documents(documents)

    if not chunks:
        raise ValueError("Aucun chunk n'a été créé à partir du PDF.")

    db = get_vector_db()
    db.add_documents(chunks)
    print("✅ Documents ajoutés dans Chroma")

    return {
        "documents_loaded": len(documents),
        "chunks_created": len(chunks),
    }


def get_llm():
    return Ollama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        timeout=300
    )


def query_rag(question: str):
    db = get_vector_db()
    results = db.similarity_search(question, k=3)

    if not results:
        return {
            "question": question,
            "answer": "Aucun contenu pertinent n'a été trouvé dans le PDF indexé.",
            "sources": []
        }

    context = "\n\n---\n\n".join([doc.page_content for doc in results])

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
        context=context,
        question=question
    )

    llm = get_llm()
    answer = llm.invoke(prompt)

    sources = [doc.metadata.get("source", "unknown") for doc in results]

    return {
        "question": question,
        "answer": answer,
        "sources": sources
    }


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    reset_db: bool = Form(False)
):
    start_time = time.time()
    print("📄 Upload reçu:", file.filename)  # 👈 AJOUT ICI

    try:
        if not file.filename:
            return {"error": "Aucun fichier fourni."}

        if not file.filename.lower().endswith(".pdf"):
            return {"error": "Seuls les fichiers PDF sont autorisés."}

        Path(DATA_PATH).mkdir(parents=True, exist_ok=True)
        

        if reset_db:
            clear_database()
            clear_data_folder()

        file_path = os.path.join(DATA_PATH, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = ingest_documents()

        with mlflow.start_run(run_name="upload-indexation"):
            mlflow.log_param("filename", file.filename)
            mlflow.log_param("model", OLLAMA_MODEL)
            mlflow.log_param("embedding_model", "HuggingFaceEmbeddings")
            mlflow.log_param("chunk_size", 800)
            mlflow.log_param("chunk_overlap", 80)
            mlflow.log_metric("documents_loaded", result["documents_loaded"])
            mlflow.log_metric("nombre_chunks", result["chunks_created"])
            mlflow.log_metric("temps_indexation", time.time() - start_time)

        return {
            "message": "PDF uploadé et indexé avec succès.",
            "filename": file.filename,
            "result": result
        }

    except Exception as e:
        return {"error": str(e)}


@app.post("/ask")
def ask_question(question: str = Form(...)):
    start_time = time.time()
    print("🔥 Question reçue:", question)

    try:
        if not question.strip():
            return {
                "success": False,
                "error": "La question ne peut pas être vide."
            }

        result = query_rag(question)

        with mlflow.start_run(run_name="question-answering"):
            mlflow.log_param("model", OLLAMA_MODEL)
            mlflow.log_param("question", question[:250])
            mlflow.log_metric("temps_reponse", time.time() - start_time)
            mlflow.log_metric("nombre_sources", len(result.get("sources", [])))

        return {
        "success": True,
        **result
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }