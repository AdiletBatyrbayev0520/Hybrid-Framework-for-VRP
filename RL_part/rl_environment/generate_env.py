import numpy as np
import pandas as pd
import argparse
import os

# Функция для генерации случайных координат городов
def generate_cities(num_cities=10):
    cities = np.random.rand(num_cities, 2) * 100  # Генерация координат от 0 до 100
    return cities

# Функция для расчета евклидова расстояния между двумя городами
def calculate_distance(city1, city2):
    return np.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)

# Главная функция
def main():
    # Парсим аргументы командной строки
    parser = argparse.ArgumentParser(description="Generate TSP Cities and Distances")
    parser.add_argument('--num_cities', type=int, help="Number of cities")
    parser.add_argument('--save_path', type=str, help="Path to save CSV files")
    args = parser.parse_args()

    num_cities = args.num_cities
    save_path = args.save_path

    # Генерация случайных городов
    cities = generate_cities(num_cities)
    distance_matrix = np.zeros((num_cities, num_cities))

    # Заполняем матрицу расстояний
    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            distance = calculate_distance(cities[i], cities[j])
            distance_matrix[i][j] = distance_matrix[j][i] = distance

    # Сохранение координат городов в CSV
    cities_df = pd.DataFrame(cities, columns=["x", "y"], index=[f"City_{i}" for i in range(num_cities)])
    cities_df.to_csv(os.path.join(save_path, "cities.csv"))

    # Сохранение матрицы расстояний в CSV
    distance_df = pd.DataFrame(distance_matrix, 
                               index=[f"City_{i}" for i in range(num_cities)], 
                               columns=[f"City_{i}" for i in range(num_cities)])
    distance_df.to_csv(os.path.join(save_path, "distances.csv"))

    print(f"Cities data saved to: {os.path.join(save_path, 'cities.csv')}")
    print(f"Distances data saved to: {os.path.join(save_path, 'distances.csv')}")

if __name__ == "__main__":
    main()