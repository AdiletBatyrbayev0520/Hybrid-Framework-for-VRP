import numpy as np
import pandas as pd
import argparse
import os
import math


def generate_cities(num_cities=10):
    
    latitudes = np.random.uniform(-90, 90, (num_cities, 1))
    
    longitudes = np.random.uniform(-180, 180, (num_cities, 1))
    
    cities = np.concatenate([latitudes, longitudes], axis=1)
    return cities


def calculate_distance(city1, city2):
    """
    Расчет расстояния между двумя точками на поверхности Земли,
    используя формулу гаверсинуса (расстояние по большой окружности)
    
    city1, city2: координаты в формате [latitude, longitude] в градусах
    возвращает: расстояние в километрах
    """
    
    R = 6371.0
    
    
    lat1 = math.radians(city1[0])
    lon1 = math.radians(city1[1])
    lat2 = math.radians(city2[0])
    lon2 = math.radians(city2[1])
    
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    
    distance = R * c
    
    return distance


def main():
    
    parser = argparse.ArgumentParser(description="Generate TSP Cities and Distances")
    parser.add_argument('--num_cities', type=int, help="Number of cities")
    parser.add_argument('--save_path', type=str, help="Path to save CSV files")
    args = parser.parse_args()

    num_cities = args.num_cities
    save_path = args.save_path

    # Создаем директорию для сохранения, если она не существует
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Created directory: {save_path}")
    
    cities = generate_cities(num_cities)
    distance_matrix = np.zeros((num_cities, num_cities))

    
    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            distance = calculate_distance(cities[i], cities[j])
            distance_matrix[i][j] = distance_matrix[j][i] = distance

    
    cities_df = pd.DataFrame(cities, columns=["latitude", "longitude"], index=[f"City_{i}" for i in range(num_cities)])
    cities_df.to_csv(os.path.join(save_path, "cities.csv"))

    
    distance_df = pd.DataFrame(distance_matrix, 
                               index=[f"City_{i}" for i in range(num_cities)], 
                               columns=[f"City_{i}" for i in range(num_cities)])
    distance_df.to_csv(os.path.join(save_path, "distances.csv"))

    print(f"Cities data saved to: {os.path.join(save_path, 'cities.csv')}")
    print(f"Distances data saved to: {os.path.join(save_path, 'distances.csv')}")

if __name__ == "__main__":
    main()