import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re

def parse_route_file(file_path):
    """Парсинг файла с маршрутом и длиной пути"""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Извлечение маршрута
    route_line = lines[1].strip()
    route = re.findall(r'City_(\d+)', route_line)
    route = [int(city) for city in route]
    
    # Извлечение длины пути
    distance_line = lines[3].strip()
    total_distance = float(distance_line.split(': ')[1])
    
    return route, total_distance

def load_cities(folder_path):
    """Загрузка координат городов из CSV-файла"""
    cities_path = os.path.join(folder_path, 'cities.csv')
    if not os.path.exists(cities_path):
        print(f"Файл {cities_path} не найден.")
        return None
    
    cities_df = pd.read_csv(cities_path, index_col=0)
    return cities_df

def analyze_routes(dp_route, dp_distance, rl_route, rl_distance):
    """Анализ различий между маршрутами"""
    # Разница в длине пути
    distance_diff = rl_distance - dp_distance
    distance_diff_percent = (distance_diff / dp_distance) * 100
    
    # Анализ порядка городов
    common_subsequences = find_common_subsequences(dp_route, rl_route)
    
    # Вывод результатов анализа
    print("АНАЛИЗ МАРШРУТОВ")
    print("-" * 50)
    print(f"Маршрут DP (Хелд-Карп): {' -> '.join(['City_' + str(city) for city in dp_route])}")
    print(f"Маршрут RL (Q-learning): {' -> '.join(['City_' + str(city) for city in rl_route])}")
    print("-" * 50)
    print(f"Длина маршрута DP: {dp_distance:.2f}")
    print(f"Длина маршрута RL: {rl_distance:.2f}")
    print(f"Разница: +{distance_diff:.2f} ({distance_diff_percent:.2f}%)")
    print("-" * 50)
    
    print("Общие подпоследовательности городов:")
    for seq in common_subsequences:
        if len(seq) > 2:  # Показываем только подпоследовательности длиннее 2
            print(f"  {' -> '.join(['City_' + str(city) for city in seq])}")
    
    return {
        "dp_distance": dp_distance,
        "rl_distance": rl_distance,
        "difference": distance_diff,
        "difference_percent": distance_diff_percent,
        "common_subsequences": common_subsequences
    }

def find_common_subsequences(route1, route2):
    """Находит общие подпоследовательности в двух маршрутах"""
    subsequences = []
    current_seq = []
    
    # Создаем словарь позиций для второго маршрута
    positions = {city: i for i, city in enumerate(route2)}
    
    for i in range(len(route1) - 1):
        current_city = route1[i]
        next_city = route1[i + 1]
        
        # Если текущий город в текущей последовательности, добавляем следующий
        if not current_seq or current_city == current_seq[-1]:
            current_seq.append(current_city)
            
            # Проверяем, является ли переход (current_city -> next_city) частью маршрута RL
            if current_city in positions and next_city in positions:
                pos_current = positions[current_city]
                pos_next = positions[next_city]
                
                # Проверяем, следуют ли города последовательно в маршруте RL
                # Учитываем циклическую природу маршрута
                if (pos_next == (pos_current + 1) % len(route2)):
                    current_seq.append(next_city)
                else:
                    # Завершаем текущую последовательность
                    if len(current_seq) > 1:
                        subsequences.append(current_seq)
                    current_seq = []
        else:
            # Завершаем текущую последовательность
            if len(current_seq) > 1:
                subsequences.append(current_seq)
            current_seq = [current_city]
    
    # Добавляем последнюю последовательность, если она не пуста
    if current_seq and len(current_seq) > 1:
        subsequences.append(current_seq)
    
    return subsequences

def visualize_routes(cities_df, dp_route, rl_route, save_path=None):
    """Визуализация маршрутов на основе координат городов"""
    if cities_df is None:
        print("Невозможно визуализировать маршруты без координат городов.")
        return
    
    plt.figure(figsize=(12, 10))
    
    # Извлечение координат
    x = cities_df['X'].values
    y = cities_df['Y'].values
    
    # Создание координат для маршрутов
    dp_x = [x[city] for city in dp_route]
    dp_y = [y[city] for city in dp_route]
    rl_x = [x[city] for city in rl_route]
    rl_y = [y[city] for city in rl_route]
    
    # Добавление замыкающего города для DP (если маршрут не замкнут)
    if dp_route[0] != dp_route[-1]:
        dp_x.append(x[dp_route[0]])
        dp_y.append(y[dp_route[0]])
    
    # Построение маршрутов
    plt.plot(dp_x, dp_y, 'b-', alpha=0.7, linewidth=2, label='DP (Хелд-Карп)')
    plt.plot(rl_x, rl_y, 'r-', alpha=0.7, linewidth=2, label='RL (Q-learning)')
    
    # Отметка городов
    plt.scatter(x, y, c='black', s=50, zorder=5)
    
    # Пометка городов
    for i, (xi, yi) in enumerate(zip(x, y)):
        plt.annotate(f'City_{i}', (xi, yi), xytext=(5, 5), textcoords='offset points')
    
    # Выделение начального города
    plt.scatter(x[0], y[0], c='green', s=100, zorder=6, label='Начальный город (City_0)')
    
    plt.title('Сравнение маршрутов алгоритмов Хелд-Карпа и Q-learning')
    plt.xlabel('X координата')
    plt.ylabel('Y координата')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Сохранение или отображение
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Визуализация сохранена в {save_path}")
    else:
        plt.show()

def create_bar_chart(dp_distance, rl_distance, save_path=None):
    """Создание столбчатой диаграммы для сравнения длин маршрутов"""
    labels = ['DP (Хелд-Карп)', 'RL (Q-learning)']
    distances = [dp_distance, rl_distance]
    colors = ['blue', 'red']
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, distances, color=colors, alpha=0.7)
    
    # Добавление значений над столбцами
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                 f'{height:.2f}', ha='center', va='bottom')
    
    plt.title('Сравнение длин маршрутов')
    plt.ylabel('Общая длина маршрута')
    plt.grid(True, axis='y', alpha=0.3)
    
    # Процентная разница
    diff_percent = ((rl_distance - dp_distance) / dp_distance) * 100
    plt.figtext(0.5, 0.01, f'Разница: {diff_percent:.2f}%', ha='center', fontsize=12)
    
    # Сохранение или отображение
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Диаграмма сохранена в {save_path}")
    else:
        plt.show()

def main():
    # Пути к файлам
    dp_route_file = "DP_part/tests/1/best_route.txt"
    rl_route_file = "RL_part/tests/1/predicted_route.txt"
    cities_folder = "RL_part/tests/1"  # Папка, содержащая cities.csv
    
    # Парсинг файлов с маршрутами
    dp_route, dp_distance = parse_route_file(dp_route_file)
    rl_route, rl_distance = parse_route_file(rl_route_file)
    
    # Загрузка координат городов
    cities_df = load_cities(cities_folder)
    
    # Анализ маршрутов
    analysis_results = analyze_routes(dp_route, dp_distance, rl_route, rl_distance)
    
    # Создание директории для результатов
    os.makedirs("analysis_results", exist_ok=True)
    
    # Визуализация маршрутов
    if cities_df is not None:
        visualize_routes(cities_df, dp_route, rl_route, save_path="analysis_results/route_comparison.png")
    
    # Создание столбчатой диаграммы
    create_bar_chart(dp_distance, rl_distance, save_path="analysis_results/distance_comparison.png")
    
    # Сохранение результатов анализа в текстовый файл
    with open("analysis_results/analysis_summary.txt", "w") as f:
        f.write("СРАВНИТЕЛЬНЫЙ АНАЛИЗ АЛГОРИТМОВ TSP\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("1. ДЛИНА МАРШРУТОВ\n")
        f.write("-" * 50 + "\n")
        f.write(f"Алгоритм Хелда-Карпа (DP): {dp_distance:.2f}\n")
        f.write(f"Double Q-learning (RL): {rl_distance:.2f}\n")
        f.write(f"Абсолютная разница: +{analysis_results['difference']:.2f}\n")
        f.write(f"Относительная разница: +{analysis_results['difference_percent']:.2f}%\n\n")
        
        f.write("2. МАРШРУТЫ\n")
        f.write("-" * 50 + "\n")
        f.write(f"DP: {' -> '.join(['City_' + str(city) for city in dp_route])}\n\n")
        f.write(f"RL: {' -> '.join(['City_' + str(city) for city in rl_route])}\n\n")
        
        f.write("3. ОБЩИЕ ПОДПОСЛЕДОВАТЕЛЬНОСТИ\n")
        f.write("-" * 50 + "\n")
        if not analysis_results['common_subsequences']:
            f.write("Не найдено значимых общих подпоследовательностей.\n")
        else:
            for i, seq in enumerate(analysis_results['common_subsequences'], 1):
                if len(seq) > 2:  # Показываем только подпоследовательности длиннее 2
                    f.write(f"Подпоследовательность {i}: {' -> '.join(['City_' + str(city) for city in seq])}\n")
        
        f.write("\n4. ВЫВОДЫ\n")
        f.write("-" * 50 + "\n")
        f.write("- Алгоритм Хелда-Карпа находит оптимальный маршрут с меньшей общей длиной.\n")
        f.write("- Double Q-learning находит приближенное решение, которое длиннее оптимального на ")
        f.write(f"{analysis_results['difference_percent']:.2f}%.\n")
        f.write("- Преимуществом RL-подхода является возможность работы с большим количеством городов.\n")
        f.write("- Качество решения RL-алгоритма можно улучшить, увеличив количество эпизодов обучения.\n")
    
    print(f"Результаты анализа сохранены в директории 'analysis_results'")

if __name__ == "__main__":
    main() 