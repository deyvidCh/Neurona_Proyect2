# app.py
import os
from typing import List

import faiss
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sentence_transformers import SentenceTransformer

from schema import QueryIn, QueryOut, Hit
from utils import construir_respuesta

# Rutas/constantes
INDEX_DIR   = "index"
INDEX_PATH  = os.path.join(INDEX_DIR, "faiss.index")
NPY_META    = os.path.join(INDEX_DIR, "meta.npy")
MODEL_NAME  = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_CACHE = "models"  # cache local del modelo HF

# FastAPI
app = FastAPI(title="Neurona pfSense", version="1.0")

# CORS abierto para pruebas / demo local (ajusta en producción)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir demo.html (UI mínima) en /demo
# También expone /static para servir archivos si hiciera falta
app.mount("/static", StaticFiles(directory=".", html=False), name="static")

@app.get("/demo")
def demo():
    # Abre http://127.0.0.1:8000/demo
    if not os.path.exists("demo.html"):
        raise HTTPException(404, "No existe demo.html en la raíz del proyecto.")
    return FileResponse("demo.html")


# ===== Carga de índice y modelo al arranque =====
if not os.path.exists(INDEX_PATH) or not os.path.exists(NPY_META):
    # Mensaje claro si falta construir el índice
    raise RuntimeError(
        "No se encontró el índice o metadatos. Ejecuta primero:  python ingest.py"
    )

# Cargar FAISS y metadatos
index = faiss.read_index(INDEX_PATH)
meta = np.load(NPY_META, allow_pickle=True)
df = pd.DataFrame.from_records(meta)
if df.empty:
    raise RuntimeError("Metadatos vacíos. Reingesta el CSV (python ingest.py).")

# Cargar modelo de embeddings
model = SentenceTransformer(MODEL_NAME, cache_folder=MODEL_CACHE)

def embed(texts: List[str]) -> np.ndarray:
    """Obtiene embeddings normalizados (para similitud coseno con IndexFlatIP)."""
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

def _load_df_and_index():
    """Recarga df/index desde disco (generados por ingest.py)."""
    if not os.path.exists(INDEX_PATH) or not os.path.exists(NPY_META):
        raise RuntimeError("Falta índice o metadatos; corre ingest.py primero.")

    _index = faiss.read_index(INDEX_PATH)
    _meta = np.load(NPY_META, allow_pickle=True)
    _df = pd.DataFrame.from_records(_meta)

    if _df.empty:
        raise RuntimeError("Metadatos vacíos tras reindex (verifica tu CSV).")

    return _df, _index


# ===== Endpoints =====
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
    """
    Busca los k intentos más cercanos a la pregunta y construye una respuesta:
    - Prioriza un intento exitoso como principal
    - Lista fallidos relacionados para evitar repetir errores
    """
    if df.empty:
        raise HTTPException(500, "Metadatos vacíos. Reingesta el CSV.")

    # Buscar en FAISS
    q_emb = embed([q.pregunta])
    scores, ids = index.search(q_emb, q.k)

    ids_row = [i for i in ids[0] if i != -1]
    scores_row = scores[0][:len(ids_row)]

    if not ids_row:
        return QueryOut(respuesta="Sin resultados.", hits=[])

    # Empaquetar hits
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
    @app.post("/reindex")
    def reindex(x_token: str = Header(default="")):
        """
        Recarga el DataFrame y el índice FAISS cargándolos desde disco.
        PROTEGIDO por header:  X-Token: <REINDEX_TOKEN>
        Flujo correcto:
          1) Edita data/intentos.csv
          2) Genera archivos:  python ingest.py  (o docker exec ... python ingest.py)
          3) Llama a POST /reindex con el header X-Token
        """
        token_ok = os.environ.get("REINDEX_TOKEN", "")
        if not token_ok or x_token != token_ok:
            raise HTTPException(status_code=401, detail="No autorizado")
    
        global df, index
        df, index = _load_df_and_index()
        return {"ok": True, "vectors": int(index.ntotal)}
    # Construir texto-respuesta
    respuesta = construir_respuesta([h.model_dump() for h in hits])
    return QueryOut(respuesta=respuesta, hits=hits)

    