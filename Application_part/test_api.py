import requests
import numpy as np
import json
import time
import matplotlib.pyplot as plt
import pandas as pd


DP_API_URL = "http://localhost:8000/solve"
RL_API_URL = "http://localhost:8001/solve"

def generate_test_matrix(n):
    """
    Генерирует тестовую матрицу расстояний размером n x n.
    
    Args:
        n: Количество городов
        
    Returns:
        Матрица расстояний
    """

    np.random.seed(42)  
    x = np.random.rand(n) * 100
    y = np.random.rand(n) * 100
    
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i][j] = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
    
    return distance_matrix.tolist()

def test_dp_api():
    """
    Тестирование API для алгоритма Хелда-Карпа (DP)
    """
    print("\nТестирование DP API (Held-Karp):")
    
    
    print("\nТест 1: Малая матрица (5 городов)")
    distance_matrix = generate_test_matrix(5)
    
    
    start_time = time.time()
    response = requests.post(
        DP_API_URL,
        json={"distance_matrix": distance_matrix}
    )
    request_time = time.time() - start_time
    
    
    if response.status_code == 200:
        result = response.json()
        print(f"Маршрут: {' -> '.join(result['route'])}")
        print(f"Длина маршрута: {result['total_distance']:.2f}")
        print(f"Время выполнения (API): {result['execution_time']:.6f} сек")
        print(f"Время запроса: {request_time:.6f} сек")
    else:
        print(f"Ошибка: {response.status_code}")
        print(response.json())
    
    
    print("\nТест 2: Средняя матрица (10 городов)")
    distance_matrix = generate_test_matrix(10)
    
    
    start_time = time.time()
    response = requests.post(
        DP_API_URL,
        json={"distance_matrix": distance_matrix}
    )
    request_time = time.time() - start_time
    
    
    if response.status_code == 200:
        result = response.json()
        print(f"Маршрут: {' -> '.join(result['route'])}")
        print(f"Длина маршрута: {result['total_distance']:.2f}")
        print(f"Время выполнения (API): {result['execution_time']:.6f} сек")
        print(f"Время запроса: {request_time:.6f} сек")
    else:
        print(f"Ошибка: {response.status_code}")
        print(response.json())
    
    
    print("\nТест 3: Максимальная матрица (20 городов)")
    distance_matrix = generate_test_matrix(20)
    
    
    start_time = time.time()
    response = requests.post(
        DP_API_URL,
        json={"distance_matrix": distance_matrix}
    )
    request_time = time.time() - start_time
    
    
    if response.status_code == 200:
        result = response.json()
        print(f"Маршрут: {' -> '.join(result['route'][:3])} ... {' -> '.join(result['route'][-3:])}")
        print(f"Длина маршрута: {result['total_distance']:.2f}")
        print(f"Время выполнения (API): {result['execution_time']:.6f} сек")
        print(f"Время запроса: {request_time:.6f} сек")
    else:
        print(f"Ошибка: {response.status_code}")
        print(response.json())
    
    
    print("\nТест 4: Слишком большая матрица (25 городов)")
    distance_matrix = generate_test_matrix(25)
    
    
    start_time = time.time()
    response = requests.post(
        DP_API_URL,
        json={"distance_matrix": distance_matrix}
    )
    request_time = time.time() - start_time
    
    
    if response.status_code != 200:
        print(f"Ожидаемая ошибка: {response.status_code}")
        print(response.json())
    else:
        print("Ошибка: API принял слишком большую матрицу!")
    
    
    print("\nТест 5: Некорректная матрица (не квадратная)")
    distance_matrix = generate_test_matrix(5)
    distance_matrix = distance_matrix[:4]  
    
    
    response = requests.post(
        DP_API_URL,
        json={"distance_matrix": distance_matrix}
    )
    
    
    if response.status_code != 200:
        print(f"Ожидаемая ошибка: {response.status_code}")
        print(response.json())
    else:
        print("Ошибка: API принял некорректную матрицу!")

def test_rl_api():
    """
    Тестирование API для RL-алгоритма
    """
    print("\nТестирование RL API:")
    
    
    print("\nТест 1: Малая матрица (5 городов)")
    distance_matrix = generate_test_matrix(5)
    
    
    start_time = time.time()
    response = requests.post(
        RL_API_URL,
        json={
            "distance_matrix": distance_matrix,
            "start_city": 0,
            "end_city": 4
        }
    )
    request_time = time.time() - start_time
    
    
    if response.status_code == 200:
        result = response.json()
        print(f"Маршрут: {' -> '.join(result['route'])}")
        print(f"Длина маршрута: {result['total_distance']:.2f}")
        print(f"Время выполнения (API): {result['execution_time']:.6f} сек")
        print(f"Время запроса: {request_time:.6f} сек")
    else:
        print(f"Ошибка: {response.status_code}")
        print(response.json())
    
    
    print("\nТест 2: Средняя матрица (10 городов)")
    distance_matrix = generate_test_matrix(10)
    
    
    start_time = time.time()
    response = requests.post(
        RL_API_URL,
        json={
            "distance_matrix": distance_matrix,
            "start_city": 0,
            "end_city": 9
        }
    )
    request_time = time.time() - start_time
    
    
    if response.status_code == 200:
        result = response.json()
        print(f"Маршрут: {' -> '.join(result['route'])}")
        print(f"Длина маршрута: {result['total_distance']:.2f}")
        print(f"Время выполнения (API): {result['execution_time']:.6f} сек")
        print(f"Время запроса: {request_time:.6f} сек")
    else:
        print(f"Ошибка: {response.status_code}")
        print(response.json())
    
    
    print("\nТест 3: Большая матрица (20 городов)")
    distance_matrix = generate_test_matrix(20)
    
    
    start_time = time.time()
    response = requests.post(
        RL_API_URL,
        json={
            "distance_matrix": distance_matrix,
            "start_city": 0,
            "end_city": 19
        }
    )
    request_time = time.time() - start_time
    
    
    if response.status_code == 200:
        result = response.json()
        print(f"Маршрут: {' -> '.join(result['route'][:3])} ... {' -> '.join(result['route'][-3:])}")
        print(f"Длина маршрута: {result['total_distance']:.2f}")
        print(f"Время выполнения (API): {result['execution_time']:.6f} сек")
        print(f"Время запроса: {request_time:.6f} сек")
    else:
        print(f"Ошибка: {response.status_code}")
        print(response.json())
    
    
    print("\nТест 4: Некорректный start_city")
    distance_matrix = generate_test_matrix(5)
    
    
    response = requests.post(
        RL_API_URL,
        json={
            "distance_matrix": distance_matrix,
            "start_city": 10,
            "end_city": 4
        }
    )
    
    
    if response.status_code != 200:
        print(f"Ожидаемая ошибка: {response.status_code}")
        print(response.json())
    else:
        print("Ошибка: API принял некорректный start_city!")

def compare_algorithms():
    """
    Сравнение алгоритмов DP и RL
    """
    print("\nСравнение алгоритмов DP и RL:")
    
    
    sizes = [5, 10, 15, 20]
    dp_times = []
    rl_times = []
    dp_distances = []
    rl_distances = []
    
    for size in sizes:
        print(f"\nТестирование матрицы размером {size}x{size}")
        distance_matrix = generate_test_matrix(size)
        
        
        response_dp = requests.post(
            DP_API_URL,
            json={"distance_matrix": distance_matrix}
        )
        
        if response_dp.status_code == 200:
            result_dp = response_dp.json()
            dp_times.append(result_dp['execution_time'])
            dp_distances.append(result_dp['total_distance'])
            print(f"DP: Время = {result_dp['execution_time']:.6f} сек, Длина = {result_dp['total_distance']:.2f}")
        else:
            print(f"DP API ошибка: {response_dp.status_code}")
            dp_times.append(None)
            dp_distances.append(None)
        
        
        response_rl = requests.post(
            RL_API_URL,
            json={
                "distance_matrix": distance_matrix,
                "start_city": 0,
                "end_city": size - 1
            }
        )
        
        if response_rl.status_code == 200:
            result_rl = response_rl.json()
            rl_times.append(result_rl['execution_time'])
            rl_distances.append(result_rl['total_distance'])
            print(f"RL: Время = {result_rl['execution_time']:.6f} сек, Длина = {result_rl['total_distance']:.2f}")
        else:
            print(f"RL API ошибка: {response_rl.status_code}")
            rl_times.append(None)
            rl_distances.append(None)
    
    
    comparison_data = {
        'Размер': sizes,
        'DP Время (сек)': dp_times,
        'RL Время (сек)': rl_times,
        'DP Длина': dp_distances,
        'RL Длина': rl_distances,
        'Разница длин (%)': [((rl - dp) / dp * 100) if dp and rl else None for dp, rl in zip(dp_distances, rl_distances)]
    }
    
    df = pd.DataFrame(comparison_data)
    print("\nСравнительная таблица:")
    print(df.to_string(index=False))
    
    
    plt.figure(figsize=(12, 5))
    
    
    plt.subplot(1, 2, 1)
    plt.plot(sizes, dp_times, 'b-o', label='DP (Held-Karp)')
    plt.plot(sizes, rl_times, 'r-o', label='RL (Q-learning)')
    plt.xlabel('Количество городов')
    plt.ylabel('Время выполнения (сек)')
    plt.title('Сравнение времени выполнения')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    
    plt.subplot(1, 2, 2)
    plt.plot(sizes, dp_distances, 'b-o', label='DP (Held-Karp)')
    plt.plot(sizes, rl_distances, 'r-o', label='RL (Q-learning)')
    plt.xlabel('Количество городов')
    plt.ylabel('Длина маршрута')
    plt.title('Сравнение длины маршрутов')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    try:
        plt.savefig('comparison_results.png')
        print("\nГрафик сохранен в файл: comparison_results.png")
    except Exception as e:
        print(f"\nОшибка при сохранении графика: {e}")
        
        import os
        current_dir = os.getcwd()
        plt.savefig(os.path.join(current_dir, 'comparison_results.png'))
        print(f"График сохранен в: {os.path.join(current_dir, 'comparison_results.png')}")

def main():
    print("Тестирование API для решения задачи коммивояжера")
    
    try:
        
        requests.get("http://localhost:8000/docs")
        dp_available = True
    except:
        print("DP API не доступен. Убедитесь, что сервер запущен на порту 8000.")
        dp_available = False
    
    try:
        
        requests.get("http://localhost:8001/docs")
        rl_available = True
    except:
        print("RL API не доступен. Убедитесь, что сервер запущен на порту 8001.")
        rl_available = False
    
    if not dp_available and not rl_available:
        print("Ни один из API не доступен. Запустите серверы и повторите попытку.")
        return
    
    
    if dp_available:
        test_dp_api()
    
    if rl_available:
        test_rl_api()
    
    if dp_available and rl_available:
        compare_algorithms()

if __name__ == "__main__":
    main() 