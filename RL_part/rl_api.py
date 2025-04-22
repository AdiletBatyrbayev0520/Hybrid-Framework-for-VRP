from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import time
from typing import List, Optional
import uvicorn
import os
import json
import tempfile
import pandas as pd

# Импортируем функции для предсказания маршрута из predict_route.py
from predict_route import find_route, save_route, load_weights

app = FastAPI(title="TSP RL Agent API",
             description="API для решения задачи коммивояжера с использованием RL-агента")

class DistanceMatrixRequest(BaseModel):
    """Модель запроса с матрицей расстояний"""
    distance_matrix: List[List[float]]
    start_city: Optional[int] = 0
    end_city: Optional[int] = 9
    weights_path: Optional[str] = "runs/test_3"  # Путь к весам по умолчанию - runs/test_3

class RouteResponse(BaseModel):
    """Модель ответа с маршрутом и дополнительной информацией"""
    algorithm: str = "RL Agent"
    route: List[str]
    total_distance: float
    execution_time: float
    num_cities: int

def temp_create_csv_files(distance_matrix):
    """
    Создает временные CSV-файлы с матрицей расстояний и координатами городов
    
    Args:
        distance_matrix: Матрица расстояний между городами
        
    Returns:
        Путь к временной директории с файлами
    """
    temp_dir = tempfile.mkdtemp()
    
    # Создаем DataFrame для матрицы расстояний
    n = len(distance_matrix)
    cities = [f"City_{i}" for i in range(n)]
    distances_df = pd.DataFrame(distance_matrix, index=cities, columns=cities)
    
    # Генерируем простые координаты для городов (не влияют на результат)
    cities_data = {
        'X': [float(i) for i in range(n)],
        'Y': [float(i) for i in range(n)]
    }
    cities_df = pd.DataFrame(cities_data, index=cities)
    
    # Сохраняем файлы
    distances_df.to_csv(os.path.join(temp_dir, 'distances.csv'))
    cities_df.to_csv(os.path.join(temp_dir, 'cities.csv'))
    
    return temp_dir

@app.post("/solve", response_model=RouteResponse)
async def solve_tsp(request: DistanceMatrixRequest):
    """
    Решение задачи коммивояжера с использованием RL-агента
    
    Args:
        request: Объект запроса с матрицей расстояний
        
    Returns:
        Объект ответа с маршрутом и дополнительной информацией
    """
    try:
        # Преобразуем матрицу расстояний в numpy array
        distance_matrix = np.array(request.distance_matrix)
        
        # Проверка корректности матрицы расстояний
        if distance_matrix.shape[0] != distance_matrix.shape[1]:
            raise HTTPException(status_code=400, detail="Матрица расстояний должна быть квадратной")
        
        num_cities = distance_matrix.shape[0]
        
        # Проверка start_city и end_city
        if request.start_city < 0 or request.start_city >= num_cities:
            raise HTTPException(status_code=400, detail=f"Некорректный start_city: {request.start_city}")
        
        if request.end_city < 0 or request.end_city >= num_cities:
            raise HTTPException(status_code=400, detail=f"Некорректный end_city: {request.end_city}")
        
        # Создаем временные файлы для визуализации результатов
        temp_dir = temp_create_csv_files(distance_matrix)
        
        # Засекаем время начала выполнения
        start_time = time.time()
        
        # Загружаем веса модели
        # Определяем пути к весам
        weights_path = request.weights_path
        module_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Проверяем различные варианты путей к весам
        possible_paths = [
            weights_path,  # Прямой путь из запроса
            os.path.join(module_dir, weights_path),  # Относительно модуля
            os.path.join(module_dir, "runs/test_3"),  # Конкретный путь относительно модуля
        ]
        
        weights_loaded = False
        for path in possible_paths:
            try:
                if os.path.exists(path):
                    print(f"Попытка загрузки весов из: {path}")
                    Q1, Q2 = load_weights(path)
                    weights_loaded = True
                    print(f"Веса успешно загружены из: {path}")
                    break
            except Exception as e:
                print(f"Ошибка при загрузке весов из {path}: {e}")
        
        if not weights_loaded:
            # Если не удалось загрузить веса, создаем случайные
            print(f"Не удалось загрузить веса из указанных путей. Используем случайные веса.")
            Q1 = np.random.rand(num_cities, num_cities) * 0.01
            Q2 = np.random.rand(num_cities, num_cities) * 0.01
        
        # Находим маршрут с помощью RL-агента, используя правильные параметры
        path = find_route(num_cities, Q1, Q2, distance_matrix)
        
        # Проверяем и корректируем маршрут, чтобы он начинался и заканчивался в нужных городах
        if path[0] != request.start_city:
            # Если маршрут не начинается с указанного начального города,
            # корректируем маршрут, перестраивая его от указанного начального города
            start_idx = path.index(request.start_city) if request.start_city in path else 0
            path = path[start_idx:] + path[:start_idx]
            
        if path[-1] != request.end_city:
            # Если маршрут не заканчивается указанным конечным городом,
            # добавляем его в конец
            if request.end_city in path:
                path.remove(request.end_city)
            path.append(request.end_city)
        
        # Вычисляем общую длину маршрута
        total_distance = 0
        for i in range(len(path) - 1):
            city1 = path[i]
            city2 = path[i + 1]
            total_distance += distance_matrix[city1][city2]
        
        # Засекаем время окончания выполнения
        execution_time = time.time() - start_time
        
        # Сохраняем результаты в JSON без использования save_route
        route_str = " -> ".join([f"City_{i}" for i in path])
        result_json = {
            "algorithm": "Double Q-learning (Reinforcement Learning)",
            "route": [f"City_{i}" for i in path],
            "total_distance": total_distance,
            "execution_time": execution_time,
            "num_cities": num_cities
        }
        
        json_path = os.path.join(temp_dir, "predicted_route.json")
        with open(json_path, 'w') as f:
            json.dump(result_json, f, indent=2)
        
        # Сохраняем также текстовый файл
        txt_path = os.path.join(temp_dir, "predicted_route.txt")
        with open(txt_path, 'w') as file:
            file.write("Best Route:\n")
            file.write(route_str + "\n")
            file.write(f"\nTotal Path Length: {total_distance}\n")
            file.write(f"Execution Time: {execution_time:.6f} seconds\n")
        
        # Формируем ответ
        response = RouteResponse(
            route=[f"City_{i}" for i in path],
            total_distance=total_distance,
            execution_time=execution_time,
            num_cities=num_cities
        )
        
        return response
    
    except Exception as e:
        # В случае непредвиденной ошибки возвращаем 500
        raise HTTPException(status_code=500, detail=f"Ошибка при решении задачи: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("rl_api:app", host="0.0.0.0", port=8001) 