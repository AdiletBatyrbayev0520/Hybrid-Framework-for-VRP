const Arrow = () => {
    return (
        <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            fill="none"
            stroke="red"
            strokeWidth={2}
            strokeLinecap="round"
            strokeLinejoin="round"
            className="h-6 w-6"
        >
            <line x1="1" y1="12" x2="20" y2="12" />
            <polyline points="15 6 20 12 15 18" />
        </svg>
    );
};


export default Arrow;