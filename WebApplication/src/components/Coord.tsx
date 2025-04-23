import {Button, Dropdown, MenuProps} from "antd";
import {useState} from "react";
import {DeleteOutlined} from "@ant-design/icons";
import {Coordinate} from "../types.ts";

const items: MenuProps['items'] = [
    {
        key: 'W',
        label: 'Warehouse',
    },
    {
        key: 'C',
        label: 'Client',
    }
];

interface Props {
    coord: Coordinate;
    handleDeleteCoordinate: (id: number) => void;
    handleTypeChange: (id: number, type: "warehouse" | "client") => void;
}


const Coord = ({coord, handleTypeChange, handleDeleteCoordinate}: Props) => {
    const [currentType, setCurrentType] = useState<'C' | 'W'>(coord.type == 'warehouse' ? 'W' : 'C');

    const onTypeClick: MenuProps['onClick'] = ({key}) => {
        if (key === 'W' || key === 'C') {
            setCurrentType(key);
            handleTypeChange(coord.id, key == 'C' ? 'client' : 'warehouse');
        }
    };

    return <div className={'border-b border-solid border-b-neutral-700 pt-2 pb-1 flex justify-between items-center'}>
        <div className="flex items-center justify-center gap-3">
            <div className={'h-16 w-16 flex items-center justify-center text-3xl cursor-pointer'}>
                <Dropdown
                    menu={{
                        items,
                        onClick: onTypeClick,
                        selectable: true,
                    }}
                    trigger={["click"]}
                >
                    <div className={`${currentType === 'W' ? 'text-[#5348F2]' : ''}`}>
                        {currentType}
                    </div>
                </Dropdown>
            </div>
            <div className="flex flex-col gap-1">
                <div className={'text-xl'}>{coord.id} address</div>
                <div className={'flex gap-2 text'}>
                    <div><span className={'text-gray-400'}>lat:</span> {Math.round(coord.lat * 1000) / 1000}</div>
                    <div><span className={'text-gray-400'}>lon:</span> {Math.round(coord.lon * 1000) / 1000}</div>
                </div>
            </div>
        </div>
        <Button color={'danger'}
                variant={'solid'}
                shape="circle"
                icon={<DeleteOutlined/>}
                className={'mr-4'}
                onClick={() => {
                    handleDeleteCoordinate(coord.id)
                }}/>

    </div>
}

export default Coord;