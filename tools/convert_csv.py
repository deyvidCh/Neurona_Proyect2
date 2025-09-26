import pandas as pd
import os

# Rutas
SRC = "data/intentos_original.csv"   # CSV que tienes ahora (con Intento,Acción,...)
DST = "data/intentos.csv"            # CSV destino con el esquema de la neurona

# Leer fuente (autodetecta separador y quita BOM)
df = pd.read_csv(SRC, sep=None, engine="python", encoding="utf-8-sig")

# Normalizar encabezados para localizar columnas aunque tengan mayúsculas/tildes
cols = {c.lower().strip(): c for c in df.columns}

def getcol(name_candidates):
    for n in name_candidates:
        key = n.lower()
        if key in cols:
            return cols[key]
    raise KeyError(f"No se encontró ninguna de las columnas {name_candidates} en el CSV origen.")

col_intento     = getcol(["intento"])
col_accion      = getcol(["acción","accion"])
col_resultado   = getcol(["resultado"])
col_explicacion = getcol(["explicación","explicacion"])

out = pd.DataFrame()
out["regla"]       = "general"  # puedes cambiarlo luego a facebook/messenger/whatsapp según el caso
out["intento"]     = df[col_intento]
out["accion"]      = df[col_accion]

# Normalizar resultado a 'exitoso' / 'falló'
def norm_result(x:str):
    s = str(x).strip().lower()
    if s.startswith("exito"):  # Exitoso, Éxito, etc.
        return "exitoso"
    if s.startswith("fall"):
        return "falló"
    return s  # deja tal cual si viene otro estado

out["resultado"]   = df[col_resultado].map(norm_result)
out["observacion"] = df[col_explicacion]
out["paso_a_paso"] = ""  # completa después si quieres
out["evidencia"]   = ""  # pon rutas a imágenes si quieres

# Guardar destino
out.to_csv(DST, index=False, encoding="utf-8")
print(f"✔ Convertido: {SRC} → {DST} ({len(out)} filas)")
