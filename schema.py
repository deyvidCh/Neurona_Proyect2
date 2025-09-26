from pydantic import BaseModel

class QueryIn(BaseModel):
    pregunta: str
    k: int = 5  # cuántos resultados traer

class Hit(BaseModel):
    regla: str
    intento: int
    accion: str
    resultado: str
    observacion: str
    paso_a_paso: str
    evidencia: str
    score: float

class QueryOut(BaseModel):
    respuesta: str
    hits: list[Hit]
