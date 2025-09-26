# app.py
import os
from typing import List

import faiss
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sentence_transformers import SentenceTransformer

from schema import QueryIn, QueryOut, Hit
from utils import construir_respuesta

# Umbral de similitud (ajustable por env)
SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", "0.35"))

# Servicios conocidos (puedes ampliar)
KNOWN_SERVICES = ["facebook", "messenger", "whatsapp", "instagram", "tiktok", "youtube", "twitter", "x"]

def detect_service(question: str) -> str | None:
    q = (question or "").lower()
    for s in KNOWN_SERVICES:
        if s in q:
            return s
    return None

def dataset_has_service(service: str) -> bool:
    """¿El dataset menciona este servicio en alguna columna clave?"""
    if not service:
        return True
    s = service.lower()
    # Busca en columnas comunes
    cols = [c for c in ["regla","accion","observacion","paso_a_paso","evidencia"] if c in df.columns]
    hay = False
    for c in cols:
        # usamos .str.contains para texto
        serie = df[c].astype(str).str.lower()
        if serie.str.contains(s, regex=False).any():
            hay = True
            break
    return hay

# === Rutas/constantes ===
INDEX_DIR   = "index"
INDEX_PATH  = os.path.join(INDEX_DIR, "faiss.index")
NPY_META    = os.path.join(INDEX_DIR, "meta.npy")
MODEL_NAME  = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_CACHE = "models"  # cache local del modelo HF
SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", "0.35"))

# === FastAPI ===
app = FastAPI(title="Neurona pfSense", version="1.0")

# CORS (en pruebas lo dejamos abierto; en prod restringe a localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ["http://127.0.0.1:8000","http://localhost:8000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir demo.html en /demo y estáticos en /static
app.mount("/static", StaticFiles(directory=".", html=False), name="static")

@app.get("/demo")
def demo():
    if not os.path.exists("demo.html"):
        raise HTTPException(404, "No existe demo.html en la raíz del proyecto.")
    return FileResponse("demo.html")

# === Carga de índice y modelo al arranque ===
if not os.path.exists(INDEX_PATH) or not os.path.exists(NPY_META):
    raise RuntimeError("No se encontró el índice o metadatos. Ejecuta primero:  python ingest.py")

index = faiss.read_index(INDEX_PATH)
meta = np.load(NPY_META, allow_pickle=True)
df = pd.DataFrame.from_records(meta)
if df.empty:
    raise RuntimeError("Metadatos vacíos. Reingesta el CSV (python ingest.py).")

model = SentenceTransformer(MODEL_NAME, cache_folder=MODEL_CACHE)

def embed(texts: List[str]) -> np.ndarray:
    """Embeddings normalizados (para similitud coseno con IndexFlatIP)."""
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

# === Helper para recargar índice/DF (usado por /reindex) ===
def _load_df_and_index():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(NPY_META):
        raise RuntimeError("Falta índice o metadatos; corre ingest.py primero.")
    _index = faiss.read_index(INDEX_PATH)
    _meta = np.load(NPY_META, allow_pickle=True)
    _df = pd.DataFrame.from_records(_meta)
    if _df.empty:
        raise RuntimeError("Metadatos vacíos tras reindex (verifica tu CSV).")
    return _df, _index

# === Endpoints ===
@app.get("/")
def root():
    return {"status": "ok", "msg": "Neurona pfSense lista. Usa POST /query"}

@app.get("/health")
def health():
    return {"ok": True, "vectors": int(index.ntotal)}

@app.get("/version")
def version():
    return {"app": "neurona-pfsense", "version": "1.0"}

@app.post("/query", response_model=QueryOut)
def query(q: QueryIn):
    if df.empty:
        raise HTTPException(500, "Metadatos vacíos. Reingesta el CSV.")
    q_emb = embed([q.pregunta])
    scores, ids = index.search(q_emb, q.k)

    ids_row = [i for i in ids[0] if i != -1]
    scores_row = scores[0][:len(ids_row)]
    
    # 1) Si no hay vecinos, mensaje claro
    if not ids_row:
        return QueryOut(
            respuesta=f"No hay evidencia en el dataset para: “{q.pregunta}”. "
                      f"Probablemente ese tema no se configuró en el proyecto.",
            hits=[]
        )
    
    # 2) Candado por servicio explícito (si pregunta dice 'instagram' pero dataset no lo tiene)
    svc = detect_service(q.pregunta)
    if svc and not dataset_has_service(svc):
        return QueryOut(
            respuesta=f"“{svc}” no se configuró en este proyecto (sin evidencia en el dataset).",
            hits=[]
        )
    
    # 3) Umbral de similitud (por si hay vecinos pero poco relevantes)
    top_score = float(scores_row[0]) if len(scores_row) else 0.0
    if top_score < SIMILARITY_THRESHOLD:
        return QueryOut(
            respuesta=(
                f"No se encontró una configuración relevante para: “{q.pregunta}”. "
                f"Parece que ese tema no se trabajó en este proyecto "
                f"(score={top_score:.3f} < umbral={SIMILARITY_THRESHOLD:.2f})."
            ),
            hits=[]
        )

    hits: List[Hit] = []
    for i, s in zip(ids_row, scores_row):
        row = df.iloc[i].to_dict()
        hits.append(Hit(
            regla=str(row.get("regla", "")),
            intento=int(row.get("intento", 0)),
            accion=str(row.get("accion", "")),
            resultado=str(row.get("resultado", "")),
            observacion=str(row.get("observacion", "")),
            paso_a_paso=str(row.get("paso_a_paso", "")),
            evidencia=str(row.get("evidencia", "")),
            score=float(s),
        ))

    respuesta = construir_respuesta([h.model_dump() for h in hits])
    return QueryOut(respuesta=respuesta, hits=hits)

@app.post("/reindex")
def reindex(x_token: str = Header(default="")):
    """
    Recarga el DataFrame y el índice FAISS desde disco.
    Protegido por header:  X-Token: <REINDEX_TOKEN>
    Flujo:
      1) Edita data/intentos.csv
      2) Genera archivos:  python ingest.py  (o docker exec ... python ingest.py)
      3) POST /reindex con header X-Token
    """
    token_ok = os.environ.get("REINDEX_TOKEN", "")
    if not token_ok or x_token != token_ok:
        raise HTTPException(status_code=401, detail="No autorizado")

    global df, index
    df, index = _load_df_and_index()
    return {"ok": True, "vectors": int(index.ntotal)}
