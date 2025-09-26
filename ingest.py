# ingest.py
import os, sys
import pandas as pd
import numpy as np

try:
    import faiss  # requiere faiss-cpu instalado
except Exception as e:
    print("[ERROR] No se pudo importar faiss. Instálalo (faiss-cpu) o usa WSL/Conda.")
    print("Detalle:", e)
    sys.exit(1)

from sentence_transformers import SentenceTransformer

DATA_CSV   = "data/intentos.csv"
INDEX_DIR  = "index"
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
NPY_META   = os.path.join(INDEX_DIR, "meta.npy")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_CACHE= "models"  # descarga local del modelo

os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(MODEL_CACHE, exist_ok=True)

def row_to_text(row):
    return (
        f"Regla: {row['regla']}\n"
        f"Intento: {row['intento']}\n"
        f"Accion: {row['accion']}\n"
        f"Resultado: {row['resultado']}\n"
        f"Observacion: {row.get('observacion','')}\n"
        f"Paso_a_paso: {row.get('paso_a_paso','')}\n"
        f"Evidencia: {row.get('evidencia','')}"
    )

def main():
    if not os.path.exists(DATA_CSV):
        print(f"[ERROR] No existe {DATA_CSV}. Crea el CSV primero.")
        sys.exit(1)

    df = pd.read_csv(DATA_CSV)
    if df.empty:
        print("[ERROR] El CSV está vacío.")
        sys.exit(1)

    df.fillna("", inplace=True)

    model = SentenceTransformer(MODEL_NAME, cache_folder=MODEL_CACHE)
    docs = df.apply(row_to_text, axis=1).tolist()
    embs = model.encode(docs, convert_to_numpy=True, normalize_embeddings=True)

    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)  # IP con vectores normalizados = coseno
    index.add(embs)

    faiss.write_index(index, INDEX_PATH)
    np.save(NPY_META, df.to_records(index=False))

    print(f"✔ Index creado: {INDEX_PATH} | vectores: {len(docs)}")

if __name__ == "__main__":
    main()
