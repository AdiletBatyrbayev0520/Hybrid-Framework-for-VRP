import numpy as np
import pandas as pd
import argparse
import os
import logging
import json
import time


def setup_logging(save_path):
    logging.basicConfig(filename=os.path.join(save_path, 'tsp_q_learning.log'), 
                        level=logging.INFO, format="%(asctime)s - %(message)s")
    logging.info("Starting Double Q-learning Algorithm")


def load_weights(weights_path):
    Q1 = np.loadtxt(os.path.join(weights_path, "q1_weights.txt"))
    Q2 = np.loadtxt(os.path.join(weights_path, "q2_weights.txt"))
    logging.info("Loaded existing weights from files")
    return Q1, Q2


def choose_action(state, Q1, Q2, epsilon, num_cities, visited):
    if np.random.rand() < epsilon:
        available_cities = [i for i in range(num_cities) if i not in visited]
        if not available_cities:
            return None
        action = np.random.choice(available_cities)
        logging.debug(f"Random action chosen: {action}")
    else:
        available_cities = [i for i in range(num_cities) if i not in visited]
        if not available_cities:
            return None
            
        q1_values = Q1[state][available_cities]
        q2_values = Q2[state][available_cities]
        
        action1 = available_cities[np.argmax(q1_values)]
        action2 = available_cities[np.argmax(q2_values)]
        
        logging.debug(f"Choosing action for state {state}: action1={action1}, action2={action2}")
        
        if np.random.rand() < 0.5:
            action = action1 
        else:
            action = action2  
    
    return action


def run_double_q_learning(num_cities, distance_matrix, num_episodes, alpha, gamma, epsilon, save_path, Q1=None, Q2=None):
    
    start_time = time.time()
    
    if Q1 is None or Q2 is None:
        Q1 = np.zeros((num_cities, num_cities))  
        Q2 = np.zeros((num_cities, num_cities))  
        logging.info("Initialized new Q-tables")
    else:
        logging.info("Using existing Q-tables for continued learning")

    
    logging.info(f"Parameters: Episodes = {num_episodes}, Alpha = {alpha}, Gamma = {gamma}, Epsilon = {epsilon}")
    
    
    for episode in range(num_episodes):
        state = np.random.choice(range(num_cities))  
        visited = [state]  
        total_reward = 0
        
        while len(visited) < num_cities:
            action = choose_action(state, Q1, Q2, epsilon, num_cities, visited)
            
            if action is None:
                break
                
            
            if action not in visited:
                visited.append(action)
                reward = -distance_matrix[state][action]  
                total_reward += reward
                
                
                next_state = action
                if np.random.rand() < 0.5:
                    Q1[state][action] += alpha * (reward + gamma * np.max(Q2[next_state]) - Q1[state][action])
                    logging.debug(f"Updated Q1[{state}][{action}] to {Q1[state][action]}")
                else:
                    Q2[state][action] += alpha * (reward + gamma * np.max(Q1[next_state]) - Q2[state][action])
                    logging.debug(f"Updated Q2[{state}][{action}] to {Q2[state][action]}")
                
                state = next_state

        logging.info(f"Episode {episode + 1}/{num_episodes}: Total reward: {total_reward}, Cities visited: {len(visited)}")

    
    best_path = [0]  
    state = 0
    visited = set(best_path)
    
    while len(best_path) < num_cities:
        action = choose_action(state, Q1, Q2, 0.0, num_cities, visited)  
        if action is None or action in visited:
            break
        best_path.append(action)
        visited.add(action)
        state = action

    
    if len(best_path) == num_cities:
        best_path.append(0)
    
    total_distance = 0
    for i in range(len(best_path) - 1):
        total_distance += distance_matrix[best_path[i]][best_path[i+1]]
    
    execution_time = time.time() - start_time
    
    logging.info(f"Best Path: {best_path}")
    logging.info(f"Total Path Length: {total_distance}")
    logging.info(f"Execution Time: {execution_time:.6f} seconds")

    route_str = " -> ".join([f"City_{i}" for i in best_path])
    
    with open(os.path.join(save_path, "best_route.txt"), "w") as file:
        file.write("Best Route:\n")
        file.write(route_str + "\n")
        file.write(f"\nTotal Path Length: {total_distance}\n")
        file.write(f"Execution Time: {execution_time:.6f} seconds\n")
        file.write(f"Training Episodes: {num_episodes}\n")

    result_json = {
        "algorithm": "Double Q-learning Training",
        "route": [f"City_{i}" for i in best_path],
        "total_distance": total_distance,
        "execution_time": execution_time,
        "num_cities": num_cities,
        "parameters": {
            "episodes": num_episodes,
            "alpha": alpha,
            "gamma": gamma,
            "epsilon": epsilon
        }
    }
    
    with open(os.path.join(save_path, "best_route.json"), "w") as json_file:
        json.dump(result_json, json_file, indent=2)

    
    np.savetxt(os.path.join(save_path, "q1_weights.txt"), Q1, fmt="%.4f")
    np.savetxt(os.path.join(save_path, "q2_weights.txt"), Q2, fmt="%.4f")
    logging.info(f"Model weights saved to 'q1_weights.txt' and 'q2_weights.txt'.")


def main():
    parser = argparse.ArgumentParser(description="Run Double Q-learning on TSP")
    parser.add_argument('--save_path', type=str, required=True, help="Path to save logs, weights, and routes")
    parser.add_argument('--weights_path', type=str, help="Path to existing weights (optional)")
    parser.add_argument('--num_episodes', type=int, default=1000, help="Number of episodes (default is 1000)")
    parser.add_argument('--alpha', type=float, default=0.1, help="Learning rate (default is 0.1)")
    parser.add_argument('--gamma', type=float, default=0.9, help="Discount factor (default is 0.9)")
    parser.add_argument('--epsilon', type=float, default=0.1, help="Exploration rate (default is 0.1)")
    args = parser.parse_args()

    save_path = args.save_path
    weights_path = args.weights_path
    num_episodes = args.num_episodes
    alpha = args.alpha
    gamma = args.gamma
    epsilon = args.epsilon

    
    cities_df = pd.read_csv(os.path.join(save_path, "cities.csv"), index_col=0)
    distance_df = pd.read_csv(os.path.join(save_path, "distances.csv"), index_col=0)

    num_cities = len(cities_df)
    distance_matrix = distance_df.values

    
    setup_logging(save_path)

    
    Q1, Q2 = None, None
    if weights_path and os.path.exists(os.path.join(weights_path, "q1_weights.txt")) and os.path.exists(os.path.join(weights_path, "q2_weights.txt")):
        Q1, Q2 = load_weights(weights_path)

    
    run_double_q_learning(num_cities, distance_matrix, num_episodes, alpha, gamma, epsilon, save_path, Q1, Q2)

if __name__ == "__main__":
    main()