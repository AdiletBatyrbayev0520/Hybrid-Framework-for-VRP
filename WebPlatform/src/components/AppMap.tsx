import {MapContainer, Marker, Popup, TileLayer, useMapEvents} from "react-leaflet";
import React, { useEffect, useRef, useState } from "react";
import {Coordinate, Road} from "../types.ts";
import L from "leaflet";
import 'leaflet-arrowheads';
import { useMap } from 'react-leaflet';

interface Props {
    coordinates: Coordinate[];
    roads: Road[];
    handleAddCoordinate: (lat: number, lon: number) => void;
}

const customIcon = (type: 'warehouse' | 'client') =>
    new L.Icon({
        iconUrl: type == 'warehouse' ? '/w-marker.svg' : '/marker.svg', // Google Charts URL for dynamic marker
        iconSize: [30, 42], // Adjust size if needed
        iconAnchor: [15, 42], // Center the anchor point
        popupAnchor: [0, -42], // Position the popup
    });

// Компонент для автоматического масштабирования карты
const AutoFitBounds: React.FC<{coordinates: Coordinate[]}> = ({ coordinates }) => {
    const map = useMap();
    
    useEffect(() => {
        if (coordinates.length > 1) {
            // Создаем границы для всех точек
            const bounds = new L.LatLngBounds(
                coordinates.map(coord => [coord.lat, coord.lon])
            );
            
            // Добавляем отступ для лучшего отображения
            const paddedBounds = bounds.pad(0.2);
            
            // Масштабируем карту, чтобы показать все точки
            map.fitBounds(paddedBounds);
        }
    }, [coordinates, map]);
    
    return null;
};

// Компонент анимации линий
interface AnimatedPolylinesProps {
    roads: Road[];
    animationEnabled: boolean;
}

const AnimatedPolylines: React.FC<AnimatedPolylinesProps> = ({ roads, animationEnabled }) => {
    const map = useMap();
    const polylinesRef = useRef<L.Polyline[]>([]);
    const animationFrameRef = useRef<number | null>(null);

    useEffect(() => {
        // Очищаем предыдущие полилинии при изменении дорог или режима
        polylinesRef.current.forEach(polyline => {
            if (map.hasLayer(polyline)) {
                map.removeLayer(polyline);
            }
        });
        polylinesRef.current = [];

        if (animationEnabled) {
            // Анимированный режим с последовательным появлением
            if (roads.length > 0) {
                roads.forEach((road: Road, index: number) => {
                    setTimeout(() => {
                        const positions: L.LatLngExpression[] = [
                            [road.start.lat, road.start.lon],
                            [road.end.lat, road.end.lon]
                        ];

                        const polyline = L.polyline(positions, {
                            color: road.highlight ? 'green' : 'blue',
                            weight: 3,
                            opacity: 0.8,
                            // Делаем линии непрерывными, убрав dashArray
                            lineCap: 'round',
                            lineJoin: 'round'
                        })
                        .arrowheads({
                            size: '10px',
                            frequency: 'endonly'
                        })
                        .addTo(map);

                        // Анимация появления
                        let step = 0;
                        const animate = () => {
                            step += 0.01;
                            if (step <= 1) {
                                polyline.setStyle({ opacity: step * 0.8 });
                                animationFrameRef.current = window.requestAnimationFrame(animate);
                            }
                        };
                        animate();

                        polylinesRef.current.push(polyline);
                    }, index * 300); // Задержка 300мс между появлением каждой линии
                });
            }
        } else {
            // Обычный режим без анимации - все линии появляются сразу
            roads.forEach((road: Road) => {
                const positions: L.LatLngExpression[] = [
                    [road.start.lat, road.start.lon],
                    [road.end.lat, road.end.lon]
                ];
                
                const polyline = L.polyline(positions, {
                    color: road.highlight ? 'green' : 'blue',
                    weight: 3,
                    opacity: 0.8
                }).addTo(map);
                
                polylinesRef.current.push(polyline);
            });
        }

        // Очистка при размонтировании
        return () => {
            if (animationFrameRef.current !== null) {
                window.cancelAnimationFrame(animationFrameRef.current);
            }
            polylinesRef.current.forEach(polyline => {
                if (map.hasLayer(polyline)) {
                    map.removeLayer(polyline);
                }
            });
        };
    }, [map, roads, animationEnabled]);

    return null;
};

const AppMap = ({coordinates, roads, handleAddCoordinate}: Props) => {
    const [animationEnabled, setAnimationEnabled] = useState<boolean>(true);

    const MapClickHandler: React.FC = () => {
        useMapEvents({
            click(e) {
                console.log(e)
                handleAddCoordinate(e.latlng.lat, e.latlng.lng);
            },
        });
        return null;
    };

    // const handleSearchLocation = async () => {
    //     try {
    //         const response = await fetch(
    //             `https://catalog.api.2gis.com/3.0/items/geocode?q=${encodeURIComponent(
    //                 search
    //             )}&key=${API_KEY}`
    //         );
    //         const data = await response.json();
    //         const point = data.result.items[0]?.point;
    //         if (point) {
    //             handleAddCoordinate(point.lat, point.lon);
    //         }
    //     } catch (err) {
    //         alert("Ошибка поиска местоположения.");
    //     }
    // };

    return (
        <div className={'flex-1 min-w-[340px]'}>
            <div className="flex justify-end mb-2">
                <button 
                    className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
                    onClick={() => setAnimationEnabled(!animationEnabled)}
                >
                    {animationEnabled ? 'Отключить анимацию' : 'Включить анимацию'}
                </button>
            </div>
            <div
                className="rounded-md overflow-hidden border-4 border-solid border-[#91898C] hover:border-[#3E3E3E] transition-all duration-200">
                <MapContainer
                    center={[43.208, 76.669]}
                    zoom={16}
                    style={{height: "400px", width: "100%"}}
                >
                    <TileLayer
                        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                    />
                    <MapClickHandler/>
                    <AnimatedPolylines roads={roads} animationEnabled={animationEnabled} />
                    <AutoFitBounds coordinates={coordinates} />
                    {coordinates.map((coord) => (
                        <Marker
                            key={coord.id}
                            position={[coord.lat, coord.lon]}
                            icon={customIcon(coord.type)}
                        >
                            <Popup>
                                Latitude: {Math.round(coord.lat * 100) / 100}, Longitude: {Math.round(coord.lon * 100) / 100} <br />
                                Id: {coord.id}
                            </Popup>
                        </Marker>
                    ))}
                </MapContainer>
            </div>
        </div>
    );
};

export default AppMap;