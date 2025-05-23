import numpy as np
import argparse
import os
import pandas as pd
import json
import time


def load_weights(weights_path):
    
    Q1 = np.loadtxt(os.path.join(weights_path, "q1_weights.txt"))
    Q2 = np.loadtxt(os.path.join(weights_path, "q2_weights.txt"))
    return Q1, Q2


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
    distance_matrix = np.zeros((num_cities, num_cities))

    for edge in edges:
        city1, city2, cost = edge
        distance_matrix[city1 - 1][city2 - 1] = cost  
        distance_matrix[city2 - 1][city1 - 1] = cost  

    return distance_matrix


def choose_action(state, Q1, Q2, epsilon, num_cities, visited):
    if np.random.rand() < epsilon:
        
        available_cities = [i for i in range(num_cities) if i not in visited]
        if not available_cities:
            return None
        return np.random.choice(available_cities)
    else:
        
        available_cities = [i for i in range(num_cities) if i not in visited]
        if not available_cities:
            return None
            
        
        q1_values = Q1[state][available_cities]
        q2_values = Q2[state][available_cities]
        
        if np.random.rand() < 0.5:
            return available_cities[np.argmax(q1_values)]
        else:
            return available_cities[np.argmax(q2_values)]


def find_route(num_cities, Q1, Q2, distance_matrix):
    epsilon = 0.1  
    best_path = [0]  
    state = 0
    visited = set(best_path)

    
    while len(best_path) < num_cities:
        next_city = choose_action(state, Q1, Q2, epsilon, num_cities, visited)
        if next_city is None:
            break
        best_path.append(next_city)
        visited.add(next_city)
        state = next_city
    
    
    best_path.append(0)
    
    return best_path


def save_route(best_path, save_path, distance_matrix, execution_time):
    
    total_distance = 0
    for i in range(len(best_path) - 1):
        total_distance += distance_matrix[best_path[i]][best_path[i+1]]
    
    
    route_str = " -> ".join([f"City_{i}" for i in best_path])
    
    with open(os.path.join(save_path, "predicted_route.txt"), "w") as file:
        file.write("Best Route:\n")
        file.write(route_str + "\n")
        file.write(f"\nTotal Path Length: {total_distance}\n")
        file.write(f"Execution Time: {execution_time:.6f} seconds\n")
    
    
    result_json = {
        "algorithm": "Double Q-learning (Reinforcement Learning)",
        "route": [f"City_{i}" for i in best_path],
        "total_distance": total_distance,
        "execution_time": execution_time,
        "num_cities": len(distance_matrix)
    }
    
    with open(os.path.join(save_path, "predicted_route.json"), "w") as json_file:
        json.dump(result_json, json_file, indent=2)
    
    return total_distance

def main():
    
    parser = argparse.ArgumentParser(description="Predict Route using Q-learning weights")
    parser.add_argument('--weights_path', type=str, required=True, help="Path to Q-learning weights")
    parser.add_argument('--input_path', type=str, required=True, help="Path to folder with cities.csv and distances.csv OR to input values file")
    parser.add_argument('--save_path', type=str, required=True, help="Path to save the predicted route")
    parser.add_argument('--csv', action='store_true', help="Use CSV files instead of input values file")
    args = parser.parse_args()

    
    Q1, Q2 = load_weights(args.weights_path)

    
    if args.csv or (os.path.isdir(args.input_path) and 
                   os.path.exists(os.path.join(args.input_path, 'cities.csv')) and 
                   os.path.exists(os.path.join(args.input_path, 'distances.csv'))):
        
        num_cities, distance_matrix = load_from_csv(args.input_path)
    else:
        
        num_cities, edges = load_input_values(args.input_path)
        distance_matrix = create_distance_matrix(num_cities, edges)

    
    start_time = time.time()
    
    
    best_path = find_route(num_cities, Q1, Q2, distance_matrix)
    
    
    execution_time = time.time() - start_time

    
    save_route(best_path, args.save_path, distance_matrix, execution_time)

if __name__ == "__main__":
    main()