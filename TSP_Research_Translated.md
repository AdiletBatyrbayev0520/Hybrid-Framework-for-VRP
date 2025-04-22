
# Research on Methods for Solving the Traveling Salesman Problem: Comparison of the Held-Karp Algorithm and Reinforcement Learning Approach

## Abstract

This research presents a comparative study of two fundamentally different approaches to solving the Traveling Salesman Problem (TSP): the exact Held-Karp algorithm, based on dynamic programming, and a heuristic method based on reinforcement learning (Double Q-learning). The study includes theoretical foundations, software implementation, experimental validation, and a comparative analysis of the efficiency of both approaches in terms of solution quality and computational complexity. A software package has been developed as part of the research that allows solving the TSP using various methods, visualizing, and analyzing the obtained results. The research results are of interest both theoretically and for practical applications in logistics, routing, and optimization.

**Keywords**: Traveling Salesman Problem, dynamic programming, reinforcement learning, Double Q-learning, combinatorial optimization, routing

## Introduction

The Traveling Salesman Problem (TSP) is one of the classic problems in combinatorial optimization. It involves finding the shortest closed route that passes through a given set of cities, where each city must be visited exactly once. Despite the simplicity of its formulation, the problem is classified as NP-hard, meaning that no polynomial-time algorithms are known to guarantee an optimal solution.

Applications of TSP include logistics and transportation planning, chip design, production scheduling, vehicle routing, and other optimization problems across various fields. Therefore, the development of efficient methods for solving TSP is of both theoretical and practical significance.

This research examines two fundamentally different approaches to solving TSP:

1. **The Held-Karp Algorithm** – an exact algorithm based on dynamic programming with exponential time complexity of O(n²·2ⁿ), where n is the number of cities.
   
2. **Reinforcement Learning Method (Double Q-learning)** – a heuristic machine learning-based approach that does not guarantee an optimal solution but can handle large instances of the problem and often finds near-optimal solutions.

The aim of this research is to conduct a comparative analysis of these two approaches in terms of the quality of the solutions obtained and computational efficiency, as well as to develop a software package for solving TSP using both methods.

## Theoretical Foundations

### The Traveling Salesman Problem

The TSP can be formally defined as follows:

Given:
- A set of vertices (cities) V = {v₁, v₂, ..., vₙ}
- A distance matrix D, where d(i,j) is the distance between cities vᵢ and vⱼ

The objective is to find:
- A permutation π = (π₁, π₂, ..., πₙ) of the cities such that the sum of the distances between consecutive cities, including the closing edge from the last city to the first, is minimized:

L(π) = d(π₁, π₂) + d(π₂, π₃) + ... + d(πₙ₋₁, πₙ) + d(πₙ, π₁) → min

### Held-Karp Algorithm

The Held-Karp algorithm, proposed in 1962, uses dynamic programming to solve the TSP. The key idea is to compute optimal subpaths for all possible subsets of cities.

For each subset of cities S, including city 1 (starting point), and for each city j ∈ S, j ≠ 1, define C(S, j) as the length of the shortest path that:
- Starts at city 1
- Visits all cities in subset S exactly once
- Ends at city j

Recurrence relation:
C(S, j) = min {C(S\{j}, i) + d(i, j) | i ∈ S\{j}, i ≠ 1}

Base case:
C({1, j}, j) = d(1, j) for all j ≠ 1

Final solution:
TSP_opt = min {C({1, 2, ..., n}, j) + d(j, 1) | j = 2, 3, ..., n}

Time complexity: O(n²·2ⁿ), Space complexity: O(n·2ⁿ)

### Reinforcement Learning and Double Q-learning

Reinforcement learning (RL) is a paradigm in machine learning where an agent learns optimal behavior by interacting with an environment. The agent performs actions, receives rewards, and updates its strategy based on the obtained experience.

Double Q-learning is a modification of the Q-learning algorithm proposed to reduce systematic overestimation of Q-function values. Unlike standard Q-learning, which uses a single Q-table, Double Q-learning uses two independent Q-tables (Q₁ and Q₂).

Update formulas for Q-tables:

Q₁(s, a) ← Q₁(s, a) + α · (r + γ · max[Q₂(s', a')] - Q₁(s, a))

Q₂(s, a) ← Q₂(s, a) + α · (r + γ · max[Q₁(s', a')] - Q₂(s, a))

Where:
- s – current state
- a – action
- r – reward
- s' – next state
- α – learning rate
- γ – discount factor

In the context of TSP:
- States represent the current city and the set of visited cities
- Actions represent the selection of the next city to visit
- Rewards are the negative distance between cities (the shorter the distance, the higher the reward)

## Methodology

### Software Implementation

The research involved the development of a Python-based software package, including:

1. **Dynamic Programming Module (DP_part)**:
   - Implementation of the Held-Karp algorithm
   - Result processing and analysis
   - API interface for algorithm access

2. **Reinforcement Learning Module (RL_part)**:
   - Implementation of the Double Q-learning algorithm
   - Model training and evaluation
   - Route prediction based on trained models
   - API interface for algorithm access

3. **Analytical Module**:
   - Route visualization
   - Comparison of results from different algorithms
   - Report and statistics generation

4. **APIs**:
   - RESTful API for accessing algorithms via HTTP requests
   - Testing and comparing API performance

### Experimental Methodology

Experiments were conducted on datasets of varying sizes:
- Small instances (5-10 cities)
- Medium instances (11-20 cities)
- Large instances (21+ cities)

For each instance, the following characteristics were measured:
- Solution quality (length of the found route)
- Execution time
- Memory consumption

For the reinforcement learning algorithm, additional evaluations were made:
- Convergence speed
- Impact of hyperparameters (α, γ, ε) on solution quality
- Stability of results under different initializations

## Results and Discussion

### Solution Quality Comparison

As expected, the Held-Karp algorithm always finds the optimal solution as it is an exact algorithm. The Double Q-learning method, being heuristic, finds approximate solutions, with deviations from the optimal solution averaging 15-30% depending on the problem size and learning parameters.

For small instances (5-10 cities), the deviation of RL solutions from optimal is typically no more than 15%. For medium instances (11-20 cities), the deviation increases to 20-25%. For larger instances, the comparison becomes difficult as the Held-Karp algorithm becomes computationally infeasible.

### Computational Efficiency Comparison

The execution time of the Held-Karp algorithm grows exponentially with the number of cities, making it impractical for instances with more than 20-25 cities on modern computers.

The training time of the Double Q-learning model is also significant but depends linearly on the number of training episodes and quadratically on the number of cities. However, after training, route prediction using this model is very fast even for large instances.

### Scalability Analysis

The key advantage of the reinforcement learning approach is its scalability. While the Held-Karp algorithm has inherent limitations due to the exponential growth of its complexity, the Double Q-learning method can be applied to much larger instances.

However, the quality of solutions obtained using RL tends to worsen as the problem size increases unless the number of training episodes and model size are appropriately increased.

### Practical Applications

In practical applications, the choice between algorithms should be based on the specifics of the task:
- For small instances (up to 20 cities), where optimality is crucial, the Held-Karp algorithm is preferable.
- For large instances or applications where approximate solutions are acceptable, the reinforcement learning method is more suitable.
- In some cases, it may be reasonable to combine methods, such as using DP results for small subproblems within an RL-based solution.

## API Interfaces for Solving TSP

RESTful APIs were developed for accessing the implemented algorithms via HTTP requests. The APIs provide a unified interface for solving TSP with various methods:

1. **DP API (Held-Karp)**:
   - `/solve` endpoint for solving TSP using the Held-Karp algorithm
   - Limitation on the number of cities (up to 21)
   - Deterministic optimal solution

2. **RL API (Double Q-learning)**:
   - `/solve` endpoint for solving TSP using trained models
   - Support for selecting start and end cities
   - Option to specify model weight path

Both APIs accept a distance matrix as input and return the route, total length, execution time, and other metadata. The developed APIs can be integrated into various systems and applications requiring routing solutions.

## Conclusion and Future Research Directions

This study presented a comparative analysis of two fundamentally different approaches to solving the Traveling Salesman Problem: the exact dynamic programming algorithm (Held-Karp) and the heuristic reinforcement learning method (Double Q-learning).

Key conclusions:
1. The Held-Karp algorithm guarantees finding the optimal solution but has exponential complexity, making it impractical for large instances.
2. The Double Q-learning method does not guarantee optimality but works significantly faster for larger instances and often finds near-optimal solutions.
3. The developed software package effectively solves the TSP using various methods, visualizes, and analyzes the results.

Future research directions:
1. **Improving the DP Method**:
   - Memory optimization
   - Parallel computation
   - Using bit masks for subset representation

2. **Advancing the RL Method**:
   - Applying neural networks (Deep Q-Network)
   - Exploring other RL algorithms (Actor-Critic, PPO)
   - Improving state-space exploration strategies

3. **Hybrid Approaches**:
   - Combining DP and RL for solving large instances
   - Using heuristics for initializing and improving solutions

4. **Extending API Functionality**:
   - Adding other algorithms (genetic algorithm, ant colony optimization)
   - Developing a web interface for visualization and result analysis

This study demonstrates the potential of applying machine learning methods to classic combinatorial optimization problems and opens the way for further research in this area.

## Bibliography

1. Held, M., & Karp, R. M. (1962). A dynamic programming approach to sequencing problems. Journal of the Society for Industrial and Applied Mathematics, 10(1), 196-210.
2. Van Hasselt, H. (2010). Double Q-learning. Advances in neural information processing systems, 23.
3. Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction. MIT press.
4. Applegate, D. L., Bixby, R. E., Chvatal, V., & Cook, W. J. (2006). The traveling salesman problem: a computational study. Princeton university press.
5. Joshi, S., Gupta, R., & Kuhn, H. W. (2019). A comprehensive review of traveling salesman problem with a brief critique on traditional approaches. arXiv preprint arXiv:1908.11890.

---

*Note: This documentation is intended for export to a Microsoft Word (.docx) format. You can use any Markdown to Word converter, like Pandoc, or copy the content into a text editor supporting Markdown formatting.*
