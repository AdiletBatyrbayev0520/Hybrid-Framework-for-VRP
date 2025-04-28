import React, {useState} from "react";
import "leaflet/dist/leaflet.css";
import {API_KEY, Coordinate, Road} from "./types.ts";
import AppMap from "./components/AppMap.tsx";
import SelectedCoordinates from "./components/SelectedCoordinates.tsx";
import Routes from "./components/Routes.tsx";
import Papa from "papaparse";

let id = 1;
const AVERAGE_SPEED_KMH = 15;
// const DP_API_URL = "http://192.168.0.163:8000/solve";
const DP_API_URL = "http://localhost:8001/solve";
const RL_API_URL = "http://localhost:8002/solve";

const App: React.FC = () => {
    const [coordinates, setCoordinates] = useState<Coordinate[]>([]);
    const [roads, setRoads] = useState<Road[]>([]);
    const [algorithm, setAlgorithm] = useState<string>("2Gis Api");
    const [vehicle, setVehicle] = useState<string>("Driving");
    const [typeOfAlgo, setTypeOfAlgo] = useState<string>("Jam");
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const [executionTime, setExecutionTime] = useState<number | null>(null);
    const [liveTimer, setLiveTimer] = useState<number>(0);
    const [timerInterval, setTimerInterval] = useState<NodeJS.Timeout | null>(null);
    const [distanceMatrix, setDistanceMatrix] = useState<number[][]>([]);
    const [totalDistance, setTotalDistance] = useState<number>(0);

    // Добавление точки
    const handleAddCoordinate = (lat: number, lon: number) => {
        setCoordinates((prev) => {
            // Находим максимальный id среди существующих координат
            const maxId = prev.length > 0 
                ? Math.max(...prev.map(coord => coord.id))
                : 0;
            
            // Устанавливаем новый id на единицу больше максимального
            const newId = maxId + 1;
            
            // Обновляем глобальную переменную id, если новый id больше
            if (newId > id) {
                id = newId + 1;
            }
            
            return [
                ...prev,
                {id: newId, lat, lon, type: "client"},
            ];
        });
    };

    // Удаление точки
    const handleDeleteCoordinate = (id: number) => {
        setCoordinates((prev) => {
            const filteredCoords = prev.filter((coord) => coord.id !== id);
            return filteredCoords.map((coord) => {
                if (coord.id > id) {
                    return { ...coord, id: coord.id - 1 };
                }
                return coord;
            });
        });
        
        setRoads((prev) => {
            const filteredRoads = prev.filter(
                (road) => road.start.id !== id && road.end.id !== id
            );
            
            return filteredRoads.map((road) => {
                const newRoad = { ...road };
                
                if (newRoad.start.id > id) {
                    newRoad.start = { 
                        ...newRoad.start, 
                        id: newRoad.start.id - 1 
                    };
                }
                
                if (newRoad.end.id > id) {
                    newRoad.end = { 
                        ...newRoad.end, 
                        id: newRoad.end.id - 1 
                    };
                }
                
                return newRoad;
            });
        });
    };

    // Изменение типа точки
    const handleTypeChange = (id: number, type: "warehouse" | "client") => {
        setCoordinates((prev) =>
            prev.map((coord) =>
                coord.id === id ? {...coord, type} : coord
            )
        );
    };

    const API2gis = async () => {
        const points = coordinates.map((coord) => ({lat: coord.lat, lon: coord.lon}));
        const requestBody = {
            points,
            sources: points.map((_, index) => index),
            targets: points.map((_, index) => index),
            transport: vehicle.toLowerCase(),
            type: typeOfAlgo.toLowerCase(),
        };

        // Создание задачи
        const createTaskResponse = await fetch(
            `https://routing.api.2gis.com/async_matrix/create_task/get_dist_matrix?key=${API_KEY}&version=2.0`,
            {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify(requestBody),
            }
        );
        const createTaskData = await createTaskResponse.json();
        if (!createTaskData?.task_id) throw new Error("Не удалось создать задачу.");

        const taskId = createTaskData.task_id;

        // Проверка статуса задачи
        let status = "TASK_IN_QUEUE";
        let resultLink = "";
        while (status !== "TASK_DONE") {
            const statusResponse = await fetch(
                `https://routing.api.2gis.com/async_matrix/result/get_dist_matrix/${taskId}?version=2.0&key=${API_KEY}`
            );
            const statusData = await statusResponse.json();
            resultLink = statusData.result_link;
            status = statusData.status;
            if (status === "TASK_CANCELED") throw new Error("Задача завершилась с ошибкой.");
            if (status === "TASK_IN_QUEUE" || status === 'TASK_IN_PROGRESS') await new Promise((resolve) => setTimeout(resolve, 5000)); // Ожидание 2 секунды
        }

        console.log(resultLink);

        // Получение результата
        const resultResponse = await fetch(resultLink);
        const resultData = await resultResponse.json();

        // Обработка результата
        if (resultData?.routes) {
            let newRoads: Road[] = resultData.routes.map((route: any) => ({
                start: coordinates[route.source_id],
                end: coordinates[route.target_id],
                distance: route.distance,
                duration: route.duration,
            }));
            setCoordinates(coordinates); 
            
            // Создаем матрицу расстояний на основе полученных данных
            const coordinatesCount = coordinates.length;
            let distanceMatrix: number[][] = Array(coordinatesCount)
                .fill(0)
                .map(() => Array(coordinatesCount).fill(0));
            
            // Заполняем матрицу расстояний из данных routes
            resultData.routes.forEach((route: any) => {
                const sourceId = route.source_id;
                const targetId = route.target_id;
                const distance = route.distance;
                
                distanceMatrix[sourceId][targetId] = distance;
                // Если у нас неориентированный граф (расстояние туда и обратно одинаковое)
                distanceMatrix[targetId][sourceId] = distance;
            });

            setDistanceMatrix(distanceMatrix);
            newRoads = newRoads
                .filter((road) => road.start.id !== road.end.id)
                .sort((a, b) => b.distance - a.distance);
            setRoads(newRoads);     
        } else {
            console.error("Маршруты не найдены.");
        }
    }


    const handleCalculateRoutesAsync = async () => {
        try {
            // Сбрасываем таймер и время перед началом нового запроса
            setExecutionTime(null);
            setLiveTimer(0);

            setIsLoading(true);

            // Очищаем предыдущий интервал таймера
            if (timerInterval) {
                clearInterval(timerInterval);
                setTimerInterval(null);
            }

            // Таймер будет обновляться каждую 0.1 секунды
            const interval = setInterval(() => {
                setLiveTimer((prev) => parseFloat((prev + 0.1).toFixed(1)));
            }, 100);
            setTimerInterval(interval);

            // Засекаем время начала выполнения
            const startTime = performance.now();

            // Выполнение алгоритма
            if (algorithm === "2Gis Api") {
                await API2gis();  // Запуск 2Gis API

            } else if (algorithm === "Held-Karp") {
                const requestBody = {
                    distance_matrix: distanceMatrix,
                };

                // Send the request to the DP API (Held-Karp)
                const response = await fetch(DP_API_URL, {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify(requestBody),
                });

                if (response.ok) {
                    const result = await response.json();
                    console.log("Result from DP API:", result);

                    // eslint-disable-next-line prefer-const
                    let {route, total_distance} = result;

                    setTotalDistance(total_distance);

                    route = route.map((city : string) => {
                        const match = city.match(/City_(\d+)/);  // Ищем цифры после "City_"
                        return match ? parseInt(match[1]) : null;  // Если совпадение найдено, извлекаем число
                    }).filter((num : number) => num !== null);  // Фильтруем null значения, если они есть

                    const newRoads: Road[] = [];
                    for (let i = 0; i < route.length - 1; i++) {
                        newRoads.push({
                            start: coordinates[route[i]],
                            end: coordinates[route[i + 1]],
                            distance: distanceMatrix[route[i]][route[i + 1]],
                            duration: (distanceMatrix[route[i]][route[i + 1]] / AVERAGE_SPEED_KMH) * 60, // Convert to minutes
                        });
                    }

                    setRoads(newRoads);
                } else {
                    console.error("Ошибка при расчете маршрутов:", response.statusText);
                    alert("Ошибка при расчете маршрутов. Попробуйте еще раз.");
                    setIsLoading(false);
                    setRoads([]);
                    setTotalDistance(0);
                }
            } else if (algorithm === "Q-learning") {
                const requestBody = {
                    distance_matrix: distanceMatrix,
                };

                // Send the request to the RL API (Q-learning)
                const response = await fetch(RL_API_URL, {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify(requestBody),
                });

                if (response.ok) {
                    const result = await response.json();
                    console.log("Result from RL API:", result);

                    let {route, total_distance} = result;

                    setTotalDistance(total_distance);

                    route = route.map((city : string) => {
                        const match = city.match(/City_(\d+)/);  
                        return match ? parseInt(match[1]) : null;  
                    }).filter((num : number) => num !== null);  

                    const newRoads: Road[] = [];
                    for (let i = 0; i < route.length - 1; i++) {
                        newRoads.push({
                            start: coordinates[route[i]],
                            end: coordinates[route[i + 1]],
                            distance: distanceMatrix[route[i]][route[i + 1]],
                            duration: (distanceMatrix[route[i]][route[i + 1]] / AVERAGE_SPEED_KMH) * 60, // Convert to minutes
                        });
                    }

                    setRoads(newRoads);
                } else {
                    console.error("Ошибка при расчете маршрутов:", response.statusText);
                    alert("Ошибка при расчете маршрутов. Попробуйте еще раз.");
                    setIsLoading(false);
                    setRoads([]);
                    setTotalDistance(0);
                }
            }

            // Засекаем время окончания
            const endTime = performance.now();
            const durationInSeconds = (endTime - startTime) / 1000;
            setExecutionTime(durationInSeconds); // Записываем результат времени выполнения
        } catch (err) {
            console.error("Ошибка при расчете маршрутов:", err);
        } finally {
            // Завершаем загрузку
            setIsLoading(false);

            // Очищаем интервал таймера после завершения запроса
            if (timerInterval) {
                clearInterval(timerInterval);
                setTimerInterval(null);
            }
        }
    };


    const handleFilesUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = e.target.files;
        if (files?.length === 2) {
            setRoads([]);
            setTotalDistance(0);
            let coordinateFile = files[0]; // Пробуем сначала первый файл
            let distanceFile = files[1];   // Пробуем второй файл

            // Сначала проверим, какой из файлов — координаты, а какой — расстояния
            const isCoordinateFile1 = coordinateFile.name.includes("cities");
            const isDistanceFile2 = distanceFile.name.includes("distances");

            // Если имя файла не совпадает, меняем местами файлы
            if (!isCoordinateFile1 || !isDistanceFile2) {
                // Поменяем местами файлы
                const temp = coordinateFile;
                coordinateFile = distanceFile;
                distanceFile = temp;
            }

            // Чтение файла с координатами (cities.csv)
            Papa.parse(coordinateFile, {
                complete: (coordinateResult) => {
                    const coordinatesData = coordinateResult.data as string[][];
                    const coordinates: Coordinate[] = coordinatesData.slice(1).map((row, index) => {
                        const lat = parseFloat(row[1]);
                        const lon = parseFloat(row[2]);

                        // Проверка на NaN для координат
                        if (isNaN(lat) || isNaN(lon)) {
                            console.warn(`Пропущена строка с некорректными координатами: ${row}`);
                            return null; // Если NaN, пропускаем эту строку
                        }

                        id = index
                        return {
                            id: index,
                            lat,
                            lon,
                            type: "client", // По умолчанию "client"
                        } as Coordinate;
                    }).filter((coord) => coord !== null); // Отфильтровываем некорректные строки

                    // Чтение файла с расстояниями (distances.csv)
                    Papa.parse(distanceFile, {
                        complete: (distanceResult) => {
                            const distanceData = distanceResult.data as string[][];
                            const distanceMatrix: number[][] = distanceData.slice(1).map((row) =>
                                row.slice(1).map((distance) => {
                                    const parsedDistance = parseFloat(distance);
                                    // Проверка на NaN для расстояний
                                    if (isNaN(parsedDistance)) {
                                        console.warn(`Пропущена строка с некорректным расстоянием: ${row}`);
                                        return NaN; // Если NaN, возвращаем NaN для пропуска
                                    }
                                    return parsedDistance;
                                })
                            ).filter((row) => row.length > 0);

                            // Отфильтровываем NaN значения в матрице расстояний
                            const filteredMatrix = distanceMatrix.map((row) =>
                                row.filter((distance) => !isNaN(distance))
                            );

                            console.log("Coordinates:", coordinates);
                            console.log("Distance Matrix:", filteredMatrix);

                            setCoordinates(coordinates); // Устанавливаем координаты
                            setDistanceMatrix(filteredMatrix); // Устанавливаем матрицу расстояний
                        },
                        header: false, // Указываем, что нет заголовков в CSV
                    });
                },
                header: false, // Указываем, что нет заголовков в CSV
            });

        } else {
            alert("Пожалуйста, загрузите оба файла.");
        }
    };

    return (
        <div className="max-w-[1400px] m-auto">
            <div className={'flex  flex-wrap m-auto gap-16 mt-16'}>
                <AppMap coordinates={coordinates}
                        roads={roads}
                        handleAddCoordinate={handleAddCoordinate}
                />
                <div>
                    {isLoading && (
                        <p className="text-yellow-400 mt-2 text-sm">
                            ⏳ Executing: {liveTimer.toFixed(1)} sec...
                        </p>
                    )}

                    {!isLoading && executionTime !== null && (
                        <p className="text-green-400 mt-2 text-sm">
                            ✅ Result: {executionTime.toFixed(2)} sec
                        </p>
                    )}

                    {!isLoading && totalDistance !== null && (
                        <p className="text-gray-400 mt-2 text-sm">
                            Total distance: {totalDistance.toFixed(2)}
                        </p>
                    )}
                    <SelectedCoordinates coordinates={coordinates}
                                         handleTypeChange={handleTypeChange}
                                         handleDeleteCoordinate={handleDeleteCoordinate}
                                         handleCalculateRoutes={handleCalculateRoutesAsync}
                                         algorithm={algorithm}
                                         vehicle={vehicle}
                                         typeOfAlgo={typeOfAlgo}
                                         setAlgorithm={setAlgorithm}
                                         setVehicle={setVehicle}
                                         setTypeOfAlgo={setTypeOfAlgo}
                                         isLoading={isLoading}
                    />
                    {
                        (algorithm === 'Held-Karp' || algorithm === 'Q-learning') && (
                            <>
                                <label
                                    htmlFor="fileInput"
                                    className="bg-blue-600 text-white p-3 rounded-lg cursor-pointer hover:bg-blue-500 transition-all duration-300 flex items-center justify-center space-x-2"
                                >
                                    <span>Загрузить CSV</span>
                                    <svg
                                        xmlns="http://www.w3.org/2000/svg"
                                        width="16"
                                        height="16"
                                        fill="currentColor"
                                        className="bi bi-cloud-upload"
                                        viewBox="0 0 16 16"
                                    >
                                        <path
                                            d="M8 0a4 4 0 0 0-4 4c0 1.016.551 1.892 1.358 2.43A5.006 5.006 0 0 0 0 10a5 5 0 0 0 5 5h6a5 5 0 0 0 4.642-3.57A3.995 3.995 0 0 0 12 14a4 4 0 0 0-4-4H8V8h4a2 2 0 1 0-2-2h-3a1 1 0 0 0-1 1v5h-.5a2 2 0 0 0-2 2V6H5z"/>
                                    </svg>
                                </label>
                                <input
                                    type="file"
                                    accept=".csv"
                                    onChange={handleFilesUpload}
                                    id="fileInput"
                                    className="hidden"
                                    multiple
                                />
                            </>
                        )
                    }
                </div>

            </div>
            <Routes roads={roads}/>
        </div>
    );
};

export default App;