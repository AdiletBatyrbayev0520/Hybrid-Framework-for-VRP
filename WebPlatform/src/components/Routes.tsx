import {Coordinate, Road} from "../types.ts";


interface MarkProps {
    coord: Coordinate;
}

const Mark = ({coord}: MarkProps) => {
    return <>
        <div
            className={'border-b border-solid border-b-neutral-700 pt-2 pb-1 flex justify-between items-center'}>
            <div className="flex items-center justify-center gap-3">
                <div className={'h-16 w-16 flex items-center justify-center text-3xl cursor-pointer'}>
                    <div className={`${coord.type === 'warehouse' ? 'text-[#5348F2]' : ''}`}>
                        {coord.type[0].toUpperCase()}
                    </div>
                </div>
                <div className="flex flex-col gap-1">
                    <div className={'text-xl'}>{coord.id} address</div>
                    <div className={'flex gap-2 text'}>
                        <div><span className={'text-gray-400'}>lat:</span> {Math.round(coord.lat * 1000) / 1000}
                        </div>
                        <div><span className={'text-gray-400'}>lon:</span> {Math.round(coord.lon * 1000) / 1000}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </>
}

interface RouteProps {
    roads: Road[];
}

const Routes = ({roads}: RouteProps) => {

    return (
        <div className="mt-16">
            <h1 className="text-3xl font-bold">Routes</h1>
            <div className="border border-solid border-[#515151] rounded-md px-2 mt-4">
                {roads.map((road: Road, index) => (
                    <div key={index} className="flex items-center justify-between flex-wrap gap-1">
                        {/* Стартовая точка */}
                        <Mark coord={road.start}/>
                        <div className="flex flex-col items-center flex-grow mx-4 relative">
                            <span className="text-red-400 text-sm">{road.distance} m.</span>
                            <div className="absolute top-1/2 left-0 right-0 h-[2px] bg-red-500 -translate-y-1/2"></div>
                            <span className="text-gray-400 text-sm">{(road.duration / 60).toFixed(2)} min.</span>
                        </div>
                        {/* Конечная точка */}
                        <Mark coord={road.end}/>
                    </div>
                ))}
            </div>
        </div>

    );
};

export default Routes;