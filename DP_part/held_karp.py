import numpy as np
import os
import pandas as pd
from itertools import combinations

def load_input_values(input_values_path):
    with open(input_values_path, 'r') as file:
        lines = file.readlines()

    num_cities = int(lines[0].strip())
    edges = []

    for line in lines[1:]:
        edge_data = list(map(int, line.split()))
        edges.append(edge_data)
    
    return num_cities, edges

def load_from_csv(folder_path):
    cities_path = os.path.join(folder_path, 'cities.csv')
    distances_path = os.path.join(folder_path, 'distances.csv')
    
    # Проверка наличия файлов
    if not os.path.exists(cities_path) or not os.path.exists(distances_path):
        raise FileNotFoundError(f"Файлы cities.csv или distances.csv не найдены в папке {folder_path}")
    
    # Загрузка координат городов
    cities_df = pd.read_csv(cities_path, index_col=0)
    
    # Загрузка матрицы расстояний
    distances_df = pd.read_csv(distances_path, index_col=0)
    
    num_cities = len(cities_df)
    
    # Преобразование матрицы расстояний в numpy array
    distance_matrix = distances_df.values
    
    return num_cities, distance_matrix

def create_distance_matrix(num_cities, edges):
    # Создаем матрицу бесконечностей
    distance_matrix = np.full((num_cities, num_cities), float('inf'))
    
    # Заполняем диагональ нулями (расстояние до самого себя)
    np.fill_diagonal(distance_matrix, 0)

    # Заполняем матрицу расстояний из рёбер
    for edge in edges:
        city1, city2, cost = edge
        # Используем индексы от 0 до n-1
        distance_matrix[city1 - 1][city2 - 1] = cost
        distance_matrix[city2 - 1][city1 - 1] = cost
    
    return distance_matrix

def held_karp(distance_matrix):
    n = len(distance_matrix)
    
    # Инициализация таблицы DP
    # dp[mask][last] = минимальная стоимость пути, заканчивающегося в last,
    # проходящего через все города в mask
    dp = {}
    parent = {}
    
    # Базовый случай: путь из начального города в каждый другой город
    for i in range(1, n):
        mask = (1 << i)
        dp[(mask, i)] = distance_matrix[0][i]
        parent[(mask, i)] = 0
    
    # Перебор всех возможных подмножеств городов
    for size in range(2, n):
        for subset in combinations(range(1, n), size):
            mask = sum(1 << i for i in subset)
            
            for last in subset:
                # Исключаем last из subset для поиска предыдущего города
                prev_mask = mask ^ (1 << last)
                
                # Находим минимальную стоимость пути
                min_cost = float('inf')
                prev_city = -1
                
                for prev in subset:
                    if prev != last:
                        cost = dp.get((prev_mask, prev), float('inf')) + distance_matrix[prev][last]
                        if cost < min_cost:
                            min_cost = cost
                            prev_city = prev
                
                if min_cost != float('inf'):
                    dp[(mask, last)] = min_cost
                    parent[(mask, last)] = prev_city
        
    # Проверяем, что город 9 не является начальным городом (0)

    target_city = n-1    
    # Находим оптимальный путь, заканчивающийся в target_city
    final_mask = (1 << n) - 2  # Все города кроме начального
    if target_city < n:
        # Исключаем target_city из маски, так как это будет последний город
        final_mask &= ~(1 << target_city)
    
    min_cost = float('inf')
    last_city = -1
    
    for i in range(1, n):
        if i != target_city:  # Исключаем target_city из промежуточных городов
            cost = dp.get((final_mask, i), float('inf')) + distance_matrix[i][target_city]
            if cost < min_cost:
                min_cost = cost
                last_city = i
    
    # Проверяем, найдено ли решение
    if min_cost == float('inf') or last_city == -1:
        path = list(range(n-1))  # Просто посещаем города по порядку
        if 0 not in path:
            path.insert(0, 0)  # Начинаем с города 0
        if target_city not in path:
            path.append(target_city)  # Заканчиваем target_city (9)
        
        # Рассчитываем длину пути
        total_distance = 0
        for i in range(len(path) - 1):
            if distance_matrix[path[i]][path[i+1]] < float('inf'):
                total_distance += distance_matrix[path[i]][path[i+1]]
            else:
                total_distance += 1000  # Штраф за отсутствующее ребро
            
        return path, total_distance
    
    # Восстанавливаем путь от начального города до предпоследнего
    path = [0]  # Начинаем с города 0
    current_mask = final_mask
    current_city = last_city
    
    while current_city != 0 and current_city > 0:
        path.append(current_city)
        prev_city = parent.get((current_mask, current_city), 0)
        current_mask ^= (1 << current_city)
        current_city = prev_city
    
    # Добавляем целевой город (9) в конец пути
    path.append(target_city)
    
    # Рассчитываем фактическую длину пути
    total_distance = 0
    for i in range(len(path) - 1):
        from_city = path[i]
        to_city = path[i+1]
        distance = distance_matrix[from_city][to_city]
        if distance < float('inf'):
            total_distance += distance
        else:
            total_distance += 1000  # Штраф
    
    return path, total_distance

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Solve TSP using Held-Karp algorithm")
    parser.add_argument('--input_path', type=str, help="Path to folder with cities.csv and distances.csv OR to input values file")
    parser.add_argument('--save_path', type=str, help="Path to save the best route")
    parser.add_argument('--csv', action='store_true', help="Use CSV files instead of input values file")
    args = parser.parse_args()
    
    # Определяем источник данных (CSV или текстовый файл)
    if args.csv or (os.path.isdir(args.input_path) and 
                    os.path.exists(os.path.join(args.input_path, 'cities.csv')) and 
                    os.path.exists(os.path.join(args.input_path, 'distances.csv'))):
        # Загружаем данные из CSV файлов
        num_cities, distance_matrix = load_from_csv(args.input_path)
    else:
        # Загружаем данные из текстового файла
        num_cities, edges = load_input_values(args.input_path)
        distance_matrix = create_distance_matrix(num_cities, edges)
    
    # Проверка на максимальное количество городов
    MAX_CITIES = 21
    if num_cities > MAX_CITIES:
        with open(os.path.join(args.save_path, "best_route.txt"), "w") as file:
            file.write(f"Ошибка: слишком много городов ({num_cities}). Максимально допустимое количество: {MAX_CITIES}.\n")
        return
    
    # Решаем задачу
    path, total_distance = held_karp(distance_matrix)
    
    # Сохраняем результат
    with open(os.path.join(args.save_path, "best_route.txt"), "w") as file:
        file.write("Best Route:\n")
        file.write(" -> ".join([f"City_{i}" for i in path]) + "\n")
        file.write(f"\nTotal Path Length: {total_distance}\n")

if __name__ == "__main__":
    main() 