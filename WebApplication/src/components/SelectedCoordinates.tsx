import {Coordinate} from "../types.ts";
import {Button, Dropdown, MenuProps} from "antd";
import Coord from "./Coord.tsx";
import React from "react";

interface Props {
    coordinates: Coordinate[];
    handleTypeChange: (id: number, type: "warehouse" | "client") => void;
    handleDeleteCoordinate: (id: number) => void;
    handleCalculateRoutes:  () => Promise<void>;
    setAlgorithm: React.Dispatch<React.SetStateAction<string>>
    setVehicle: React.Dispatch<React.SetStateAction<string>>
    setTypeOfAlgo: React.Dispatch<React.SetStateAction<string>>
    algorithm: string;
    vehicle: string;
    typeOfAlgo: string;
    isLoading: boolean;
}

const algoMenuItems: MenuProps['items'] = [
    {
        key: '2Gis Api',
        label: '2Gis Api'
    },
    {
        key: 'Held-Karp',
        label: 'Held-Karp'
    },
    {
        key: 'Q-learning',
        label: 'Q-learning'
    }
]


const vehicleMenuItems: MenuProps['items'] = [
    {
        key: 'Truck',
        label: 'Truck'
    },
    {
        key: 'Driving',
        label: 'Driving'
    },
    {
        key: 'Taxi',
        label: 'Taxi'
    }
]

const algoTypeMenuItems: MenuProps['items'] = [
    {
        key: 'Jam',
        label: 'Jam'
    },
    {
        key: 'Statistics',
        label: 'Statistics'
    },
    {
        key: 'Shortest',
        label: 'Shortest'
    }
]

const SelectedCoordinates = ({
                                 coordinates,
                                 handleTypeChange,
                                 handleDeleteCoordinate,
                                 handleCalculateRoutes,
                                 setAlgorithm,
                                 setVehicle,
                                 setTypeOfAlgo,
                                 vehicle,
                                 typeOfAlgo,
                                 algorithm,
                                 isLoading,
                             }: Props) => {


    const onAlgoClick: MenuProps['onClick'] = ({key}) => {
        setAlgorithm(key);
    };

    const onVehicleClick: MenuProps['onClick'] = ({key}) => {
        setVehicle(key);
    };

    const onAlgoTypeClick: MenuProps['onClick'] = ({key}) => {
        setTypeOfAlgo(key);
    };


    return (
        <div className="flex-1">
            <div className="flex flex-col gap-4 mb-2">
                <div className={'flex justify-between flex-wrap gap-2'}>
                    <h1 className="text-3xl font-bold">Coordinates</h1>
                    <Button type={'primary'}
                            size={'large'}
                            className={'min-w-40 h-9'}
                            disabled={coordinates.length === 0}
                            onClick={() => handleCalculateRoutes()}
                            loading={isLoading}
                    >SEND</Button>
                </div>
                <div className={'flex justify-between flex-wrap gap-2'}>
                    <Dropdown
                        menu={{
                            items: algoMenuItems,
                            onClick: onAlgoClick,
                            selectable: true,
                        }}
                        trigger={["click"]}
                    >
                        <div className={`bg-[#282828] p-2 rounded w-48 pl-4 h-full`}>
                            {algorithm}
                        </div>
                    </Dropdown>
                    <Dropdown
                        menu={{
                            items: vehicleMenuItems,
                            onClick: onVehicleClick,
                            selectable: true,
                        }}
                        trigger={["click"]}
                    >
                        <div className={`bg-[#282828] p-2 rounded w-48 pl-4 h-full`}>
                            {vehicle}
                        </div>
                    </Dropdown>
                    <Dropdown
                        menu={{
                            items: algoTypeMenuItems,
                            onClick: onAlgoTypeClick,
                            selectable: true,
                        }}
                        trigger={["click"]}
                    >
                        <div className={`bg-[#282828] p-2 rounded w-48 pl-4 h-full`}>
                            {typeOfAlgo}
                        </div>
                    </Dropdown>
                </div>

            </div>

            <div className="border border-solid border-[#515151] rounded-md px-2 max-h-[400px] overflow-y-auto" style={{ maxHeight: 'calc(4 * 88px)' }}>
                {
                    coordinates.map((coordinate, i) => {
                        return <Coord key={i}
                                      coord={coordinate}
                                      handleDeleteCoordinate={handleDeleteCoordinate}
                                      handleTypeChange={handleTypeChange}/>
                    })
                }
            </div>

        </div>
    );
};

export default SelectedCoordinates;