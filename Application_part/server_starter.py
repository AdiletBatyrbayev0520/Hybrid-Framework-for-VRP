import subprocess
import time
import os
import signal
import sys

def is_port_in_use(port):
    """Проверка, занят ли указанный порт"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def start_servers():
    print("Запуск API-серверов для задачи коммивояжера...")
    
    
    if is_port_in_use(8000):
        print("Порт 8000 уже занят. Возможно, DP API уже запущен.")
    
    if is_port_in_use(8001):
        print("Порт 8001 уже занят. Возможно, RL API уже запущен.")
    
    
    print("\nЗапуск DP API сервера (Held-Karp)...")
    dp_process = subprocess.Popen(
        ["python", "-m", "uvicorn", "dp_api:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd="../DP_part",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    
    time.sleep(1)
    if dp_process.poll() is not None:
        print("Ошибка запуска DP API сервера:")
        print(dp_process.stderr.read())
    else:
        print("DP API сервер запущен на http://localhost:8000")
    
    
    print("\nЗапуск RL API сервера (Q-learning)...")
    rl_process = subprocess.Popen(
        ["python", "-m", "uvicorn", "rl_api:app", "--host", "0.0.0.0", "--port", "8001"],
        cwd="../RL_part",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    
    time.sleep(1)
    if rl_process.poll() is not None:
        print("Ошибка запуска RL API сервера:")
        print(rl_process.stderr.read())
    else:
        print("RL API сервер запущен на http://localhost:8001")
    
    print("\nОба сервера запущены. Доступ к API:")
    print("DP API (Held-Karp): http://localhost:8000/docs")
    print("RL API (Q-learning): http://localhost:8001/docs")
    print("\nДля остановки серверов нажмите Ctrl+C")
    
    
    def signal_handler(sig, frame):
        print("\nОстановка серверов...")
        dp_process.terminate()
        rl_process.terminate()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    
    try:
        dp_process.wait()
        rl_process.wait()
    except KeyboardInterrupt:
        print("\nОстановка серверов...")
        dp_process.terminate()
        rl_process.terminate()

if __name__ == "__main__":
    start_servers() 