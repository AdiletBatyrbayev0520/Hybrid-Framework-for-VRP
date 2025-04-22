import numpy as np
import os
import pandas as pd
import json
import time
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
    
    
    if not os.path.exists(cities_path) or not os.path.exists(distances_path):
        raise FileNotFoundError(f"Файлы cities.csv или distances.csv не найдены в папке {folder_path}")
    
    
    cities_df = pd.read_csv(cities_path, index_col=0)
    
    
    distances_df = pd.read_csv(distances_path, index_col=0)
    
    num_cities = len(cities_df)
    
    
    distance_matrix = distances_df.values
    
    return num_cities, distance_matrix

def create_distance_matrix(num_cities, edges):
    
    distance_matrix = np.full((num_cities, num_cities), float('inf'))
    
    
    np.fill_diagonal(distance_matrix, 0)

    
    for edge in edges:
        city1, city2, cost = edge
        
        distance_matrix[city1 - 1][city2 - 1] = cost
        distance_matrix[city2 - 1][city1 - 1] = cost
    
    return distance_matrix

def held_karp(distance_matrix):
    n = len(distance_matrix)
    
    
    
    
    dp = {}
    parent = {}
    
    
    for i in range(1, n):
        dp[(1 << i, i)] = distance_matrix[0][i]
        parent[(1 << i, i)] = 0  
    
    
    for size in range(2, n):
        for subset in combinations(range(1, n), size):
            
            bits = 0
            for bit in subset:
                bits |= 1 << bit
            
            
            for last in subset:
                prev = bits & ~(1 << last)  
                
                
                dp[(bits, last)] = float('inf')
                
                
                for j in subset:
                    if j == last:
                        continue
                    
                    
                    if (prev, j) not in dp:
                        continue
                    
                    
                    val = dp[(prev, j)] + distance_matrix[j][last]
                    
                    
                    if val < dp[(bits, last)]:
                        dp[(bits, last)] = val
                        parent[(bits, last)] = j  
    
    
    bits = (1 << n) - 2  
    min_cost = float('inf')
    last = -1
    
    
    for i in range(1, n):
        if (bits, i) in dp:  
            val = dp[(bits, i)] + distance_matrix[i][0]
            if val < min_cost:
                min_cost = val
                last = i
    
    
    if min_cost == float('inf') or last == -1:
        path = list(range(n))  
        path.append(0)  
        
        
        total_distance = 0
        for i in range(len(path) - 1):
            total_distance += distance_matrix[path[i]][path[i+1]]
        
        return path, total_distance
    
    
    path = []
    
    
    cur = last
    cur_bits = bits
    
    
    while cur != 0:
        path.append(cur)
        if (cur_bits, cur) not in parent:
            
            path = list(range(n))
            path.append(0)
            
            
            total_distance = 0
            for i in range(len(path) - 1):
                total_distance += distance_matrix[path[i]][path[i+1]]
            
            return path, total_distance
        
        next_cur = parent[(cur_bits, cur)]
        cur_bits = cur_bits & ~(1 << cur)
        cur = next_cur
    
    
    path.append(0)  
    path.reverse()  
    path.append(0)  
    
    
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
    
    
    if args.csv or (os.path.isdir(args.input_path) and 
                    os.path.exists(os.path.join(args.input_path, 'cities.csv')) and 
                    os.path.exists(os.path.join(args.input_path, 'distances.csv'))):
        
        num_cities, distance_matrix = load_from_csv(args.input_path)
    else:
        
        num_cities, edges = load_input_values(args.input_path)
        distance_matrix = create_distance_matrix(num_cities, edges)
    
    
    MAX_CITIES = 21
    if num_cities > MAX_CITIES:
        with open(os.path.join(args.save_path, "best_route.txt"), "w") as file:
            file.write(f"Ошибка: слишком много городов ({num_cities}). Максимально допустимое количество: {MAX_CITIES}.\n")
        return
    
    
    start_time = time.time()
    
    
    path, total_distance = held_karp(distance_matrix)
    
    
    execution_time = time.time() - start_time
    
    
    route_str = " -> ".join([f"City_{i}" for i in path])
    
    
    with open(os.path.join(args.save_path, "best_route.txt"), "w") as file:
        file.write("Best Route:\n")
        file.write(route_str + "\n")
        file.write(f"\nTotal Path Length: {total_distance}\n")
        file.write(f"Execution Time: {execution_time:.6f} seconds\n")
    
    
    result_json = {
        "algorithm": "Held-Karp (Dynamic Programming)",
        "route": [f"City_{i}" for i in path],
        "total_distance": total_distance,
        "execution_time": execution_time,
        "num_cities": num_cities
    }
    
    with open(os.path.join(args.save_path, "best_route.json"), "w") as json_file:
        json.dump(result_json, json_file, indent=2)

if __name__ == "__main__":
    main() 