# Neurona pfSense (local)
- Ingesta CSV de intentos → embeddings (MiniLM) → FAISS.
- Responde preguntas devolviendo el intento exitoso y fallidos relacionados.

## Uso rápido
pip install -r requirements.txt
python tools/convert_csv.py # si partes de intentos_original.csv
python tools/label_regla.py # para rellenar la columna 'regla'
python ingest.py
uvicorn app:app --reload --port 8000
Probar: POST /query con `{ "pregunta": "...", "k": 5 }`
