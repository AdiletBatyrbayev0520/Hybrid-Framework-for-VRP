from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import time
from typing import List, Dict, Any
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

from held_karp import held_karp

app = FastAPI(title="TSP Held-Karp API",
             description="API для решения задачи коммивояжера с использованием алгоритма Хелда-Карпа")
origins = [
    "*",  
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)
class DistanceMatrixRequest(BaseModel):
    """Модель запроса с матрицей расстояний"""
    distance_matrix: List[List[float]]

class RouteResponse(BaseModel):
    """Модель ответа с маршрутом и дополнительной информацией"""
    algorithm: str = "Held-Karp (Dynamic Programming)"
    route: List[str]
    total_distance: float
    execution_time: float
    num_cities: int

@app.post("/solve", response_model=RouteResponse)
async def solve_tsp(request: DistanceMatrixRequest):
    """
    Решение задачи коммивояжера с использованием алгоритма Хелда-Карпа
    
    Args:
        request: Объект запроса с матрицей расстояний
        
    Returns:
        Объект ответа с маршрутом и дополнительной информацией
    """
    try:
        
        distance_matrix = np.array(request.distance_matrix)
        
        
        if distance_matrix.shape[0] != distance_matrix.shape[1]:
            raise HTTPException(status_code=400, detail="Матрица расстояний должна быть квадратной")
        
        num_cities = distance_matrix.shape[0]
        
        
        MAX_CITIES = 21
        if num_cities > MAX_CITIES:
            raise HTTPException(
                status_code=400, 
                detail=f"Слишком много городов ({num_cities}). Максимально допустимое количество: {MAX_CITIES}"
            )
        
        
        start_time = time.time()
        
        
        path, total_distance = held_karp(distance_matrix)
        
        
        execution_time = time.time() - start_time
        
        
        response = RouteResponse(
            route=[f"City_{i}" for i in path],
            total_distance=total_distance,
            execution_time=execution_time,
            num_cities=num_cities
        )
        
        return response
    
    except Exception as e:
        
        raise HTTPException(status_code=500, detail=f"Ошибка при решении задачи: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("dp_api:app", host="0.0.0.0", port=8000) 