import {StrictMode} from 'react'
import {createRoot} from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import {ConfigProvider} from "antd";
import {darkTheme} from "./theme.ts";

createRoot(document.getElementById('root')!).render(
    <StrictMode>
        <ConfigProvider theme={darkTheme}>
            <App/>
        </ConfigProvider>
    </StrictMode>,
)
