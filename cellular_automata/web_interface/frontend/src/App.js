import React from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import { SimulationProvider } from "./context/SimulationContext";
import Header from "./components/Header";
import ControlPanel from "./components/ControlPanel";
import MapInterface from "./components/MapInterface";
import ResultsPanel from "./components/ResultsPanel";
import Documentation from "./pages/Documentation";
import "./styles/main.css";

// Main Simulation View
const SimulationView = () => (
	<div className="app">
		<Header />
		<main className="main-container">
			<aside className="control-panel-container">
				<ControlPanel />
			</aside>

			<section className="content-area">
				<MapInterface />
				<ResultsPanel />
			</section>
		</main>
	</div>
);

// App with Routing
const App = () => {
	return (
		<SimulationProvider>
			<Routes>
				<Route path="/" element={<SimulationView />} />
				<Route path="/simulation" element={<SimulationView />} />
				<Route path="/help" element={<Documentation />} />
				<Route path="/docs" element={<Documentation />} />
				<Route path="*" element={<Navigate to="/" replace />} />
			</Routes>
		</SimulationProvider>
	);
};

export default App;
