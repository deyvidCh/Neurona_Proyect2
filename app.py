import os
import faiss
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from schema import QueryIn, QueryOut, Hit
from utils import construir_respuesta

INDEX_DIR   = "index"
INDEX_PATH  = os.path.join(INDEX_DIR, "faiss.index")
NPY_META    = os.path.join(INDEX_DIR, "meta.npy")
MODEL_NAME  = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_CACHE = "models"

app = FastAPI(title="Neurona pfSense", version="1.0")

# CORS abierto para pruebas locales / demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Validaciones al arranque
if not os.path.exists(INDEX_PATH) or not os.path.exists(NPY_META):
    raise RuntimeError("No se encontró el índice. Ejecuta primero:  python ingest.py")

index = faiss.read_index(INDEX_PATH)
meta = np.load(NPY_META, allow_pickle=True)
df = pd.DataFrame.from_records(meta)

if df.empty:
    raise RuntimeError("Metadatos vacíos. Reingesta el CSV.")

model = SentenceTransformer(MODEL_NAME, cache_folder=MODEL_CACHE)

def embed(texts: list[str]) -> np.ndarray:
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

@app.get("/health")
def health():
    return {"ok": True, "vectors": int(index.ntotal)}

@app.get("/version")
def version():
    return {"app": "neurona-pfsense", "version": "1.0"}

@app.get("/")
def root():
    return {"status": "ok", "msg": "Neurona pfSense lista. Usa POST /query"}

@app.post("/query", response_model=QueryOut)
def query(q: QueryIn):
    q_emb = embed([q.pregunta])
    scores, ids = index.search(q_emb, q.k)

    ids = [i for i in ids[0] if i != -1]
    scores = scores[0][:len(ids)]

    hits = []
    for i, s in zip(ids, scores):
        row = df.iloc[i].to_dict()
        hits.append(Hit(
            regla=str(row.get("regla","")),
            intento=int(row.get("intento",0)),
            accion=str(row.get("accion","")),
            resultado=str(row.get("resultado","")),
            observacion=str(row.get("observacion","")),
            paso_a_paso=str(row.get("paso_a_paso","")),
            evidencia=str(row.get("evidencia","")),
            score=float(s),
        ))

    if not hits:
        return QueryOut(respuesta="Sin resultados.", hits=[])

    respuesta = construir_respuesta([h.model_dump() for h in hits])
    return QueryOut(respuesta=respuesta, hits=hits)
