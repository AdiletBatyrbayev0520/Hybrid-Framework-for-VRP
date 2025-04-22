import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import json

def parse_route_file(file_path):
    """Парсинг файла с маршрутом и длиной пути"""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Извлечение маршрута
        route_line = lines[1].strip()
        route = re.findall(r'City_(\d+)', route_line)
        route = [int(city) for city in route]
        
        # Извлечение длины пути
        distance_line = None
        execution_time = None
        
        for line in lines:
            if "Total Path Length" in line:
                distance_line = line.strip()
            elif "Execution Time" in line:
                execution_time_line = line.strip()
                execution_time = float(execution_time_line.split(': ')[1].split(' ')[0])
        
        if distance_line:
            total_distance = float(distance_line.split(': ')[1])
        else:
            print(f"Предупреждение: Не найдена информация о длине пути в файле {file_path}")
            total_distance = None
        
        # Попытка загрузить данные из JSON, если он существует
        json_path = file_path.replace('.txt', '.json')
        json_data = None
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as jf:
                    json_data = json.load(jf)
            except:
                print(f"Предупреждение: Не удалось загрузить JSON-данные из {json_path}")
        
        return route, total_distance, execution_time, json_data
    
    except Exception as e:
        print(f"Ошибка при парсинге файла маршрута {file_path}: {e}")
        return None, None, None, None

def load_cities(folder_path):
    """Загрузка координат городов из CSV-файла"""
    cities_path = os.path.join(folder_path, 'cities.csv')
    if not os.path.exists(cities_path):
        print(f"Файл {cities_path} не найден.")
        return None
    
    cities_df = pd.read_csv(cities_path, index_col=0)
    return cities_df

def analyze_routes(dp_route, dp_distance, dp_time, rl_route, rl_distance, rl_time):
    """Анализ различий между маршрутами"""
    # Разница в длине пути
    distance_diff = rl_distance - dp_distance
    distance_diff_percent = (distance_diff / dp_distance) * 100
    
    # Разница во времени выполнения
    if dp_time and rl_time:
        time_diff = rl_time - dp_time
        time_ratio = rl_time / dp_time if dp_time > 0 else float('inf')
    else:
        time_diff = None
        time_ratio = None
    
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
    
    if dp_time and rl_time:
        print("-" * 50)
        print(f"Время выполнения DP: {dp_time:.6f} сек")
        print(f"Время выполнения RL: {rl_time:.6f} сек")
        print(f"Разница: {time_diff:.6f} сек (RL в {time_ratio:.2f}x {'быстрее' if time_ratio < 1 else 'медленнее'})")
    
    print("-" * 50)
    
    print("Общие подпоследовательности городов:")
    for seq in common_subsequences:
        if len(seq) > 2:  # Показываем только подпоследовательности длиннее 2
            print(f"  {' -> '.join(['City_' + str(city) for city in seq])}")
    
    return {
        "dp_distance": dp_distance,
        "rl_distance": rl_distance,
        "distance_difference": distance_diff,
        "distance_difference_percent": distance_diff_percent,
        "dp_time": dp_time,
        "rl_time": rl_time,
        "time_difference": time_diff,
        "time_ratio": time_ratio,
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

def create_comparison_charts(analysis_results, save_path=None):
    """Создание сравнительных диаграмм"""
    # Создаем фигуру с двумя подграфиками
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Данные для диаграмм
    labels = ['DP (Хелд-Карп)', 'RL (Q-learning)']
    distances = [analysis_results["dp_distance"], analysis_results["rl_distance"]]
    times = [analysis_results["dp_time"], analysis_results["rl_time"]]
    colors = ['blue', 'red']
    
    # 1. Диаграмма сравнения длин маршрутов
    bars1 = ax1.bar(labels, distances, color=colors, alpha=0.7)
    ax1.set_title('Сравнение длин маршрутов')
    ax1.set_ylabel('Общая длина маршрута')
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Добавление значений над столбцами
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{height:.2f}', ha='center', va='bottom')
    
    # Процентная разница для маршрутов
    diff_percent = analysis_results["distance_difference_percent"]
    ax1.figtext(0.25, 0.01, f'Разница: {diff_percent:.2f}%', ha='center', fontsize=12)
    
    # 2. Диаграмма сравнения времени выполнения
    if times[0] and times[1]:
        bars2 = ax2.bar(labels, times, color=colors, alpha=0.7)
        ax2.set_title('Сравнение времени выполнения')
        ax2.set_ylabel('Время (секунды)')
        ax2.grid(True, axis='y', alpha=0.3)
        
        # Логарифмическая шкала, если времена сильно различаются
        if max(times) / min(times) > 100:
            ax2.set_yscale('log')
            ax2.set_ylabel('Время (секунды, лог. шкала)')
        
        # Добавление значений над столбцами
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height * 1.05,
                    f'{height:.4f}', ha='center', va='bottom')
        
        # Кратность разницы во времени
        time_ratio = analysis_results["time_ratio"]
        faster_text = "быстрее" if time_ratio < 1 else "медленнее"
        ax2.figtext(0.75, 0.01, f'RL в {abs(time_ratio):.2f}x {faster_text}', ha='center', fontsize=12)
    else:
        ax2.text(0.5, 0.5, 'Нет данных о времени выполнения', 
                ha='center', va='center', fontsize=14)
    
    plt.tight_layout()
    
    # Сохранение или отображение
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Диаграммы сохранены в {save_path}")
    else:
        plt.show()

def main():
    # Пути к файлам
    dp_route_file = "DP_part/tests/1/best_route.txt"
    rl_route_file = "RL_part/tests/1/predicted_route.txt"
    cities_folder = "RL_part/tests/1"  # Папка, содержащая cities.csv
    
    # Парсинг файлов с маршрутами
    dp_route, dp_distance, dp_time, dp_json = parse_route_file(dp_route_file)
    rl_route, rl_distance, rl_time, rl_json = parse_route_file(rl_route_file)
    
    # Проверка успешности парсинга
    if not all([dp_route, dp_distance, rl_route, rl_distance]):
        print("Ошибка: Не удалось корректно загрузить данные маршрутов.")
        return
    
    # Загрузка координат городов
    cities_df = load_cities(cities_folder)
    
    # Анализ маршрутов
    analysis_results = analyze_routes(dp_route, dp_distance, dp_time, rl_route, rl_distance, rl_time)
    
    # Создание директории для результатов
    os.makedirs("analysis_results", exist_ok=True)
    
    # Визуализация маршрутов
    if cities_df is not None:
        visualize_routes(cities_df, dp_route, rl_route, save_path="analysis_results/route_comparison.png")
    
    # Создание сравнительных диаграмм
    create_comparison_charts(analysis_results, save_path="analysis_results/performance_comparison.png")
    
    # Сохранение результатов анализа в текстовый файл
    with open("analysis_results/analysis_summary.txt", "w") as f:
        f.write("СРАВНИТЕЛЬНЫЙ АНАЛИЗ АЛГОРИТМОВ TSP\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("1. ДЛИНА МАРШРУТОВ\n")
        f.write("-" * 50 + "\n")
        f.write(f"Алгоритм Хелда-Карпа (DP): {dp_distance:.2f}\n")
        f.write(f"Double Q-learning (RL): {rl_distance:.2f}\n")
        f.write(f"Абсолютная разница: +{analysis_results['distance_difference']:.2f}\n")
        f.write(f"Относительная разница: +{analysis_results['distance_difference_percent']:.2f}%\n\n")
        
        if dp_time and rl_time:
            f.write("2. ВРЕМЯ ВЫПОЛНЕНИЯ\n")
            f.write("-" * 50 + "\n")
            f.write(f"Алгоритм Хелда-Карпа (DP): {dp_time:.6f} сек\n")
            f.write(f"Double Q-learning (RL): {rl_time:.6f} сек\n")
            f.write(f"Разница: {analysis_results['time_difference']:.6f} сек\n")
            faster_text = "быстрее" if analysis_results['time_ratio'] < 1 else "медленнее"
            f.write(f"RL в {abs(analysis_results['time_ratio']):.2f}x {faster_text} чем DP\n\n")
        
        f.write(f"{'3' if dp_time and rl_time else '2'}. МАРШРУТЫ\n")
        f.write("-" * 50 + "\n")
        f.write(f"DP: {' -> '.join(['City_' + str(city) for city in dp_route])}\n\n")
        f.write(f"RL: {' -> '.join(['City_' + str(city) for city in rl_route])}\n\n")
        
        f.write(f"{'4' if dp_time and rl_time else '3'}. ОБЩИЕ ПОДПОСЛЕДОВАТЕЛЬНОСТИ\n")
        f.write("-" * 50 + "\n")
        if not analysis_results['common_subsequences']:
            f.write("Не найдено значимых общих подпоследовательностей.\n")
        else:
            for i, seq in enumerate(analysis_results['common_subsequences'], 1):
                if len(seq) > 2:  # Показываем только подпоследовательности длиннее 2
                    f.write(f"Подпоследовательность {i}: {' -> '.join(['City_' + str(city) for city in seq])}\n")
        
        f.write(f"\n{'5' if dp_time and rl_time else '4'}. ВЫВОДЫ\n")
        f.write("-" * 50 + "\n")
        f.write("- Алгоритм Хелда-Карпа находит оптимальный маршрут с меньшей общей длиной.\n")
        f.write("- Double Q-learning находит приближенное решение, которое длиннее оптимального на ")
        f.write(f"{analysis_results['distance_difference_percent']:.2f}%.\n")
        
        if dp_time and rl_time:
            if analysis_results['time_ratio'] < 1:
                f.write(f"- Double Q-learning работает в {1/analysis_results['time_ratio']:.2f}x быстрее, ")
                f.write("но с потерей качества решения.\n")
            else:
                f.write(f"- Алгоритм Хелда-Карпа работает в {analysis_results['time_ratio']:.2f}x быстрее ")
                f.write("и находит оптимальное решение.\n")
        
        f.write("- Преимуществом RL-подхода является возможность работы с большим количеством городов.\n")
        f.write("- Качество решения RL-алгоритма можно улучшить, увеличив количество эпизодов обучения.\n")
    
    # Сохранение результатов анализа в JSON
    json_results = {
        "dp": {
            "route": [f"City_{city}" for city in dp_route],
            "total_distance": dp_distance,
            "execution_time": dp_time,
            "json_data": dp_json
        },
        "rl": {
            "route": [f"City_{city}" for city in rl_route],
            "total_distance": rl_distance,
            "execution_time": rl_time,
            "json_data": rl_json
        },
        "comparison": {
            "distance_difference": analysis_results['distance_difference'],
            "distance_difference_percent": analysis_results['distance_difference_percent'],
            "time_difference": analysis_results['time_difference'],
            "time_ratio": analysis_results['time_ratio']
        }
    }
    
    with open("analysis_results/analysis_summary.json", "w") as jf:
        json.dump(json_results, jf, indent=2)
    
    print(f"Результаты анализа сохранены в директории 'analysis_results'")

if __name__ == "__main__":
    main() 