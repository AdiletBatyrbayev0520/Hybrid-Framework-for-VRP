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
        dp[(1 << i, i)] = distance_matrix[0][i]
        parent[(1 << i, i)] = 0  # Родитель - начальный город
    
    # Перебор всех возможных подмножеств городов
    for size in range(2, n):
        for subset in combinations(range(1, n), size):
            # Преобразуем подмножество в битовую маску
            bits = 0
            for bit in subset:
                bits |= 1 << bit
            
            # Рассчитываем стоимость для каждого последнего города
            for last in subset:
                prev = bits & ~(1 << last)  # Исключаем последний город
                
                # Инициализируем минимальную стоимость
                dp[(bits, last)] = float('inf')
                
                # Рассматриваем все возможные предпоследние города
                for j in subset:
                    if j == last:
                        continue
                    
                    # Проверяем, что предыдущее состояние существует
                    if (prev, j) not in dp:
                        continue
                    
                    # Рассчитываем стоимость пути
                    val = dp[(prev, j)] + distance_matrix[j][last]
                    
                    # Обновляем, если найден лучший путь
                    if val < dp[(bits, last)]:
                        dp[(bits, last)] = val
                        parent[(bits, last)] = j  # Сохраняем предка для восстановления пути
    
    # Находим оптимальный цикл (возвращение в начальный город)
    bits = (1 << n) - 2  # Все города кроме начального (0)
    min_cost = float('inf')
    last = -1
    
    # Находим лучший последний город перед возвращением в начальный
    for i in range(1, n):
        if (bits, i) in dp:  # Проверяем, что состояние существует
            val = dp[(bits, i)] + distance_matrix[i][0]
            if val < min_cost:
                min_cost = val
                last = i
    
    # Если нет решения, возвращаем простой маршрут
    if min_cost == float('inf') or last == -1:
        path = list(range(n))  # Посещаем города по порядку
        path.append(0)  # Возвращаемся в начало
        
        # Считаем длину простого пути
        total_distance = 0
        for i in range(len(path) - 1):
            total_distance += distance_matrix[path[i]][path[i+1]]
        
        return path, total_distance
    
    # Восстановление пути
    path = []
    
    # Восстанавливаем путь из таблицы предков
    cur = last
    cur_bits = bits
    
    # Добавляем города в обратном порядке
    while cur != 0:
        path.append(cur)
        if (cur_bits, cur) not in parent:
            # Если что-то пошло не так, используем простой путь
            path = list(range(n))
            path.append(0)
            
            # Считаем длину простого пути
            total_distance = 0
            for i in range(len(path) - 1):
                total_distance += distance_matrix[path[i]][path[i+1]]
            
            return path, total_distance
        
        next_cur = parent[(cur_bits, cur)]
        cur_bits = cur_bits & ~(1 << cur)
        cur = next_cur
    
    # Добавляем начальный и конечный город
    path.append(0)  # Добавляем начальный город
    path.reverse()  # Переворачиваем путь
    path.append(0)  # Добавляем конечный город (возвращаемся в начало)
    
    # Считаем длину найденного пути
    total_distance = 0
    for i in range(len(path) - 1):
        total_distance += distance_matrix[path[i]][path[i+1]]
    
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