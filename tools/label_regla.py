import pandas as pd
import re

SRC = "data/intentos.csv"
DST = "data/intentos.csv"  # sobrescribe

df = pd.read_csv(SRC, encoding="utf-8-sig")
for c in df.columns:
    if c.strip().lower() == "accion":
        col_accion = c
        break
else:
    raise SystemExit("No se encontró la columna 'accion' en el CSV.")

def etiquetar_regla(txt:str) -> str:
    s = str(txt).lower()
    if any(k in s for k in ["messenger", "fbsbx", "edge-chat", "gateway.messenger"]):
        return "messenger"
    if any(k in s for k in ["whatsapp", "wa.me", "mmg.whatsapp", "graph.whatsapp"]):
        return "whatsapp"
    if any(k in s for k in ["facebook", "fbcdn", "facebook.com"]):
        return "facebook"
    return "general"

if "regla" not in [c.strip().lower() for c in df.columns]:
    df.insert(0, "regla", "")

# localizar nombre real de columna 'regla'
col_regla = next((c for c in df.columns if c.strip().lower()=="regla"), "regla")

df[col_regla] = df.apply(
    lambda r: r[col_regla] if str(r[col_regla]).strip() else etiquetar_regla(r[col_accion]),
    axis=1
)

df.to_csv(DST, index=False, encoding="utf-8")
print("✔ Etiquetado de 'regla' completado y guardado.")
