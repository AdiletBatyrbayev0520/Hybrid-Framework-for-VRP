Traveling Salesman Problem (TSP) with Double Q-learning

This project implements the Traveling Salesman Problem (TSP) solution using Double Q-learning. The agent learns to visit all cities in a given graph and returns to the starting city, with the goal of minimizing the total travel distance.

Features
	•	Double Q-learning: Uses two Q-tables to reduce overestimation bias in action-value estimates.
	•	Exploration vs. Exploitation: Uses ε-greedy strategy for balancing exploration of new cities and exploitation of learned paths.
	•	City Visits: Ensures all cities are visited exactly once before returning to the starting city.
	•	Logging and Visualization: Logs the learning process, action selections, Q-value updates, and the best route.

Files
	•	double_q_learning.py: Main script to run the Double Q-learning algorithm on the TSP.
	•	generate_tsp_data.py: Generates random city coordinates and distance matrices, saved as CSV files.
	•	cities.csv: CSV file containing the coordinates of the cities.
	•	distances.csv: CSV file containing the pairwise distances between the cities.
	•	best_route.txt: File containing the best route found by the algorithm.
	•	q1_weights.txt & q2_weights.txt: Files containing the learned Q-values for the two Q-tables.

Requirements
	•	Python 3.x
	•	numpy
	•	pandas

You can install the necessary dependencies using pip:

pip install numpy pandas

How to Use

1. Generating TSP Data

Before running the Double Q-learning algorithm, generate the cities and their distances using the generate_tsp_data.py script. This script creates the required cities.csv and distances.csv files.

python generate_tsp_data.py <num_cities> <save_path>

	•	num_cities: Number of cities for the TSP (e.g., 10).
	•	save_path: Directory where the CSV files will be saved.

Example:

python generate_tsp_data.py 10 /path/to/save

This will generate two files:
	•	cities.csv: Contains the coordinates of the cities.
	•	distances.csv: Contains the pairwise distances between cities.

2. Running Double Q-learning

Once the data is generated, you can run the Double Q-learning algorithm to solve the TSP. The double_q_learning.py script will learn the best route based on the generated data.

python double_q_learning.py <save_path> <num_episodes> <alpha> <gamma>

	•	save_path: Directory where the data and results will be saved (the same path used in step 1).
	•	num_episodes: Number of training episodes (default: 1000).
	•	alpha: Learning rate (default: 0.1).
	•	gamma: Discount factor (default: 0.9).

Example:

python double_q_learning.py /path/to/save 1000 0.1 0.9

This will:
	•	Train the agent using Double Q-learning to find the shortest route.
	•	Save the best route in best_route.txt.
	•	Save the learned Q-values in q1_weights.txt and q2_weights.txt.
	•	Log the learning process in tsp_q_learning.log.

3. Output
	•	best_route.txt: Contains the best route found by the algorithm.
	•	q1_weights.txt: The learned Q-table for the first Q-value function.
	•	q2_weights.txt: The learned Q-table for the second Q-value function.
	•	tsp_q_learning.log: Log file with detailed information on the learning process, including action selections, updates, and episode rewards.

Example Output in best_route.txt:

Best Route:
City_0 -> City_5 -> City_3 -> City_7 -> City_1 -> City_4 -> City_9 -> City_8 -> City_2 -> City_6 -> City_0

This route represents the shortest path that visits all cities exactly once and returns to the starting city (City_0).

Hyperparameters
	•	num_episodes: The number of episodes the agent will train for. A higher number of episodes will allow the agent to learn better but will take longer to compute.
	•	alpha: The learning rate, which determines how much the agent updates the Q-values. A small value means slow learning, while a larger value means faster learning.
	•	gamma: The discount factor, which determines the importance of future rewards. A higher value means the agent will focus more on future rewards.

Logging

The training process is logged in tsp_q_learning.log. It contains:
	•	Episode information.
	•	Total reward achieved in each episode.
	•	Q-table updates during learning.
	•	Best route found by the agent.

Conclusion

This project provides a solution to the Traveling Salesman Problem using Double Q-learning. By exploring the balance between exploration and exploitation, the agent learns to find an optimal route that visits all cities and returns to the start.

For further improvements, consider tuning the hyperparameters or implementing other reinforcement learning algorithms like DQN (Deep Q-Network) for more complex environments.

Feel free to modify or extend the project as needed! If you have any questions or issues, please feel free to ask for help.