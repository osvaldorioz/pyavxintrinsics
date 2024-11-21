from fastapi import FastAPI
import avx_intr
import time
from pydantic import BaseModel
from typing import List
import json

app = FastAPI()

# Definir el modelo para la matriz
class Matrix(BaseModel):
    matrix: List[List[float]]

# Definir el modelo para el vector
class VectorF(BaseModel):
    vector: List[float]

# Definir el modelo para el vector
class VectorI(BaseModel):
    vector: List[int]

@app.post("/avx")
async def intrisics(a: VectorF,
                    b: VectorF):
    start = time.time()

    # Crear vectores de ejemplo
    #a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    #b = [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]

    # Sumar vectores usando AVX
    result = avx_intr.add_vectors_avx(a.vector, b.vector)
    
    end = time.time()

    var1 = end - start

    j1 = {
        "Time taken in seconds": var1,
        "vector a": a.vector,
        "vector b": b.vector,
        "resultado": result
    }
    jj = json.dumps(j1)

    return jj