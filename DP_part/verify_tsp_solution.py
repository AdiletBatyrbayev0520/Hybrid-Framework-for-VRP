#!/usr/bin/env python3
import os
import re
import sys
import pandas as pd
import numpy as np
import argparse
import json

def parse_route_file(route_file_path):
    """
    Парсинг файла с маршрутом.
    Возвращает список городов в маршруте, общую длину пути и время выполнения.
    """
    try:
        with open(route_file_path, 'r') as f:
            lines = f.readlines()
        
        # Должно быть не менее 3 строк (заголовок, маршрут, пустая строка, длина)
        if len(lines) < 3:
            print(f"Ошибка: Файл маршрута {route_file_path} имеет неверный формат.")
            return None, None, None
        
        # Извлекаем маршрут
        route_line = lines[1].strip()
        route = re.findall(r'City_(\d+)', route_line)
        route = [int(city) for city in route]
        
        # Ищем строку с длиной пути и временем выполнения
        distance_line = None
        execution_time = None
        
        for line in lines:
            if "Total Path Length" in line or "Total path length" in line:
                distance_line = line.strip()
            elif "Execution Time" in line:
                execution_time_line = line.strip()
                # Извлекаем время выполнения, учитывая, что может быть "seconds" в конце
                execution_time = float(execution_time_line.split(':')[-1].strip().split(' ')[0])
        
        # Извлекаем длину пути
        if distance_line:
            total_distance = float(distance_line.split(':')[-1].strip())
        else:
            print("Ошибка: Не найдена информация о длине пути.")
            total_distance = None
        
        # Проверяем наличие JSON-файла с результатами
        json_path = route_file_path.replace('.txt', '.json')
        json_data = None
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as jf:
                    json_data = json.load(jf)
                # Если в JSON есть информация о времени выполнения, используем её
                if json_data and 'execution_time' in json_data:
                    execution_time = json_data['execution_time']
            except:
                print(f"Предупреждение: Не удалось загрузить JSON-данные из {json_path}")
        
        return route, total_distance, execution_time
    
    except Exception as e:
        print(f"Ошибка при парсинге файла маршрута: {e}")
        return None, None, None

def load_distance_matrix(matrix_file_path):
    """
    Загрузка матрицы расстояний из CSV-файла.
    """
    try:
        # Проверка на существование файла
        if not os.path.exists(matrix_file_path):
            print(f"Ошибка: Файл {matrix_file_path} не найден.")
            return None
        
        # Загрузка матрицы расстояний
        distances_df = pd.read_csv(matrix_file_path, index_col=0)
        distance_matrix = distances_df.values
        
        return distance_matrix
    
    except Exception as e:
        print(f"Ошибка при загрузке матрицы расстояний: {e}")
        return None

def verify_route(route, distance_matrix):
    """
    Проверка маршрута:
    1. Маршрут начинается и заканчивается в городе 0
    2. Все города посещены ровно один раз (кроме города 0)
    3. Расчет общей длины маршрута
    """
    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "calculated_distance": 0
    }
    
    # Проверка наличия маршрута
    if not route:
        results["valid"] = False
        results["errors"].append("Маршрут отсутствует.")
        return results
    
    # Проверка начального и конечного города
    if route[0] != 0:
        results["valid"] = False
        results["errors"].append(f"Маршрут должен начинаться с City_0, но начинается с City_{route[0]}.")
    
    if route[-1] != 0:
        results["valid"] = False
        results["errors"].append(f"Маршрут должен заканчиваться в City_0, но заканчивается в City_{route[-1]}.")
    
    # Подсчет количества городов
    n = len(distance_matrix)
    city_count = {}
    for city in route:
        if city not in city_count:
            city_count[city] = 0
        city_count[city] += 1
    
    # Проверка, что каждый город посещен ровно один раз (кроме города 0)
    for i in range(n):
        if i == 0:
            # Город 0 должен быть посещен ровно дважды (в начале и в конце)
            if city_count.get(i, 0) != 2:
                results["valid"] = False
                results["errors"].append(f"City_0 должен быть посещен ровно 2 раза (начало и конец), но посещен {city_count.get(i, 0)} раз.")
        else:
            # Все остальные города должны быть посещены ровно один раз
            if city_count.get(i, 0) != 1:
                results["valid"] = False
                results["errors"].append(f"City_{i} должен быть посещен ровно 1 раз, но посещен {city_count.get(i, 0)} раз.")
    
    # Проверка, что все города из матрицы расстояний посещены
    missing_cities = []
    for i in range(n):
        if i != 0 and i not in city_count:
            missing_cities.append(i)
    
    if missing_cities:
        results["valid"] = False
        results["errors"].append(f"Следующие города не были посещены: {', '.join([f'City_{c}' for c in missing_cities])}")
    
    # Вычисление общей длины маршрута
    total_distance = 0
    for i in range(len(route) - 1):
        city1 = route[i]
        city2 = route[i + 1]
        # Проверка на корректность индексов
        if city1 >= n or city2 >= n:
            results["valid"] = False
            results["errors"].append(f"Неверный индекс города: City_{max(city1, city2)} выходит за пределы матрицы расстояний ({n}x{n}).")
            continue
        
        distance = distance_matrix[city1][city2]
        total_distance += distance
    
    results["calculated_distance"] = total_distance
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Проверка решения задачи коммивояжера")
    parser.add_argument("--route", type=str, required=True, help="Путь к файлу с маршрутом (best_route.txt)")
    parser.add_argument("--matrix", type=str, required=True, help="Путь к файлу с матрицей расстояний (distances.csv)")
    parser.add_argument("--json", action="store_true", help="Сохранить результаты проверки в JSON-формате")
    args = parser.parse_args()
    
    # Загрузка данных
    route, reported_distance, execution_time = parse_route_file(args.route)
    distance_matrix = load_distance_matrix(args.matrix)
    
    if route is None or distance_matrix is None:
        print("Ошибка при загрузке данных.")
        sys.exit(1)
    
    # Проверка решения
    results = verify_route(route, distance_matrix)
    
    # Вывод результатов
    print("\n=== РЕЗУЛЬТАТЫ ПРОВЕРКИ РЕШЕНИЯ TSP ===\n")
    
    print(f"Маршрут: {' -> '.join([f'City_{city}' for city in route])}")
    
    if results["valid"]:
        print("\n✓ Маршрут валиден.")
        
        # Проверка точности расчета длины маршрута
        if reported_distance is not None:
            calculated = results["calculated_distance"]
            difference = abs(calculated - reported_distance)
            rel_diff = (difference / reported_distance) * 100 if reported_distance != 0 else float('inf')
            
            print(f"\nДлина маршрута:")
            print(f"  Указанная в файле: {reported_distance:.6f}")
            print(f"  Рассчитанная:      {calculated:.6f}")
            print(f"  Разница:           {difference:.6f} ({rel_diff:.6f}%)")
            
            if rel_diff < 0.01:
                print("\n✓ Длина маршрута рассчитана верно.")
            else:
                print("\n⚠ Есть расхождение в расчете длины маршрута.")
        
        # Вывод информации о времени выполнения
        if execution_time is not None:
            print(f"\nВремя выполнения: {execution_time:.6f} секунд")
    else:
        print("\n✗ Маршрут невалиден. Обнаружены следующие проблемы:")
        for error in results["errors"]:
            print(f"  - {error}")
    
    if results["warnings"]:
        print("\nПредупреждения:")
        for warning in results["warnings"]:
            print(f"  - {warning}")
    
    # Сохранение результатов в JSON-файл
    if args.json:
        output_json = {
            "route_validity": {
                "valid": results["valid"],
                "errors": results["errors"],
                "warnings": results["warnings"]
            },
            "route_data": {
                "route": [f"City_{city}" for city in route],
                "reported_distance": reported_distance,
                "calculated_distance": results["calculated_distance"],
                "execution_time": execution_time
            },
            "comparison": {
                "distance_difference": abs(reported_distance - results["calculated_distance"]) if reported_distance else None,
                "distance_difference_percent": (abs(reported_distance - results["calculated_distance"]) / reported_distance * 100) if reported_distance and reported_distance != 0 else None
            }
        }
        
        # Сохраняем результаты в JSON-файл
        output_path = os.path.join(os.path.dirname(args.route), "verification_results.json")
        with open(output_path, "w") as jf:
            json.dump(output_json, jf, indent=2)
        print(f"\nРезультаты проверки сохранены в {output_path}")

if __name__ == "__main__":
    main() 