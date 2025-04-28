export interface Coordinate {
    id: number;
    lat: number;
    lon: number;
    type: "warehouse" | "client";
}

export interface Road {
    start: Coordinate;
    end: Coordinate;
    distance: number;
    duration: number;
    highlight?: boolean;
}


export const API_KEY = "20309b3b-a341-4363-89ae-81a622d8daf4";
