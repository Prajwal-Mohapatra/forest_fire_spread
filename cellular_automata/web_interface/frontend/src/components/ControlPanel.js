import React, { useEffect } from "react";
import { useSimulation } from "../context/SimulationContext";

const ControlPanel = () => {
	const { state, loadAvailableDates, runSimulation, updateWeatherParam, setSimulationHours, toggleMLPrediction, clearIgnitionPoints } = useSimulation();

	// Load available dates when component mounts
	useEffect(() => {
		loadAvailableDates();
	}, [loadAvailableDates]);

	// Get values from state
	const { availableDates, selectedDate, weatherParams, simulationHours, useMLPrediction, ignitionPoints, isLoading } = state;

	// Handle date selection
	const handleDateChange = (e) => {
		const { value } = e.target;
		// Dispatch the date selection action
		const { dispatch } = useSimulation();
		dispatch({ type: "SELECT_DATE", payload: value });
	};

	// Handle ML toggle
	const handleMLToggle = () => {
		toggleMLPrediction();
	};

	return (
		<div className="control-panel">
			<div className="card">
				<div className="card-header">
					<h2 className="card-title">
						<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="currentColor" viewBox="0 0 16 16">
							<path d="M8 4a.5.5 0 0 1 .5.5V6a.5.5 0 0 1-1 0V4.5A.5.5 0 0 1 8 4zM3.732 5.732a.5.5 0 0 1 .707 0l.915.914a.5.5 0 1 1-.708.708l-.914-.915a.5.5 0 0 1 0-.707zM2 10a.5.5 0 0 1 .5-.5h1.586a.5.5 0 0 1 0 1H2.5A.5.5 0 0 1 2 10zm9.5 0a.5.5 0 0 1 .5-.5h1.5a.5.5 0 0 1 0 1H12a.5.5 0 0 1-.5-.5zm.754-4.246a.389.389 0 0 0-.527-.02L7.547 9.31a.91.91 0 1 0 1.302 1.258l3.434-4.297a.389.389 0 0 0-.029-.518z" />
							<path
								fill-rule="evenodd"
								d="M0 10a8 8 0 1 1 15.547 2.661c-.442 1.253-1.845 1.602-2.932 1.25C11.309 13.488 9.475 13 8 13c-1.474 0-3.31.488-4.615.911-1.087.352-2.49.003-2.932-1.25A7.988 7.988 0 0 1 0 10zm8-7a7 7 0 0 0-6.603 9.329c.203.575.923.876 1.68.63C4.397 12.533 6.358 12 8 12s3.604.532 4.923.96c.757.245 1.477-.056 1.68-.631A7 7 0 0 0 8 3z"
							/>
						</svg>
						Simulation Settings
					</h2>
				</div>

				<div className="form-group">
					<label className="form-label">Date</label>
					<select className="form-control" value={selectedDate} onChange={handleDateChange} disabled={availableDates.length === 0 || isLoading}>
						{availableDates.length === 0 && <option value="">Loading dates...</option>}

						{availableDates.map((date) => (
							<option key={date.value} value={date.value}>
								{date.label}
							</option>
						))}
					</select>
				</div>

				<div className="form-group">
					<label className="form-label">Simulation Hours</label>
					<div className="range-slider">
						<div className="range-slider-header">
							<span>Duration</span>
							<span className="range-slider-value">{simulationHours} hours</span>
						</div>
						<input type="range" min="1" max="24" value={simulationHours} onChange={(e) => setSimulationHours(e.target.value)} disabled={isLoading} />
					</div>
				</div>

				<div className="form-group">
					<div className="checkbox">
						<input type="checkbox" id="use-ml" checked={useMLPrediction} onChange={handleMLToggle} disabled={isLoading} />
						<label htmlFor="use-ml">Use ML Prediction</label>
					</div>
				</div>
			</div>

			<div className="card">
				<div className="card-header">
					<h2 className="card-title">
						<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="currentColor" viewBox="0 0 16 16">
							<path d="M5 8a1 1 0 1 1-2 0 1 1 0 0 1 2 0zm4 0a1 1 0 1 1-2 0 1 1 0 0 1 2 0zm3 1a1 1 0 1 0 0-2 1 1 0 0 0 0 2z" />
							<path d="m2.165 15.803.02-.004c1.83-.363 2.948-.842 3.468-1.105A9.06 9.06 0 0 0 8 15c4.418 0 8-3.134 8-7s-3.582-7-8-7-8 3.134-8 7c0 1.76.743 3.37 1.97 4.6a10.437 10.437 0 0 1-.524 2.318l-.003.011a10.722 10.722 0 0 1-.244.637c-.079.186.074.394.273.362a21.673 21.673 0 0 0 .693-.125zm.8-3.108a1 1 0 0 0-.287-.801C1.618 10.83 1 9.468 1 8c0-3.192 3.004-6 7-6s7 2.808 7 6c0 3.193-3.004 6-7 6a8.06 8.06 0 0 1-2.088-.272 1 1 0 0 0-.711.074c-.387.196-1.24.57-2.634.893a10.97 10.97 0 0 0 .398-2z" />
						</svg>
						Weather Parameters
					</h2>
				</div>

				<div className="form-group">
					<div className="range-slider">
						<div className="range-slider-header">
							<span>Wind Direction</span>
							<span className="range-slider-value">{weatherParams.wind_direction}°</span>
						</div>
						<input type="range" min="0" max="360" value={weatherParams.wind_direction} onChange={(e) => updateWeatherParam("wind_direction", e.target.value)} disabled={isLoading} />
					</div>
				</div>

				<div className="form-group">
					<div className="range-slider">
						<div className="range-slider-header">
							<span>Wind Speed</span>
							<span className="range-slider-value">{weatherParams.wind_speed} km/h</span>
						</div>
						<input type="range" min="0" max="50" value={weatherParams.wind_speed} onChange={(e) => updateWeatherParam("wind_speed", e.target.value)} disabled={isLoading} />
					</div>
				</div>

				<div className="form-group">
					<div className="range-slider">
						<div className="range-slider-header">
							<span>Temperature</span>
							<span className="range-slider-value">{weatherParams.temperature}°C</span>
						</div>
						<input type="range" min="10" max="50" value={weatherParams.temperature} onChange={(e) => updateWeatherParam("temperature", e.target.value)} disabled={isLoading} />
					</div>
				</div>

				<div className="form-group">
					<div className="range-slider">
						<div className="range-slider-header">
							<span>Relative Humidity</span>
							<span className="range-slider-value">{weatherParams.relative_humidity}%</span>
						</div>
						<input type="range" min="10" max="90" value={weatherParams.relative_humidity} onChange={(e) => updateWeatherParam("relative_humidity", e.target.value)} disabled={isLoading} />
					</div>
				</div>
			</div>

			<div className="form-actions">
				<button className="btn btn-primary" onClick={runSimulation} disabled={isLoading || ignitionPoints.length === 0}>
					<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
						<path d="M8 15A7 7 0 1 1 8 1a7 7 0 0 1 0 14zm0 1A8 8 0 1 0 8 0a8 8 0 0 0 0 16z" />
						<path d="M6.271 5.055a.5.5 0 0 1 .52.038l3.5 2.5a.5.5 0 0 1 0 .814l-3.5 2.5A.5.5 0 0 1 6 10.5v-5a.5.5 0 0 1 .271-.445z" />
					</svg>
					Run Simulation
				</button>

				<button className="btn btn-secondary" onClick={clearIgnitionPoints} disabled={isLoading || ignitionPoints.length === 0}>
					<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
						<path d="M2.5 1a1 1 0 0 0-1 1v1a1 1 0 0 0 1 1H3v9a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2V4h.5a1 1 0 0 0 1-1V2a1 1 0 0 0-1-1H10a1 1 0 0 0-1-1H7a1 1 0 0 0-1 1H2.5zm3 4a.5.5 0 0 1 .5.5v7a.5.5 0 0 1-1 0v-7a.5.5 0 0 1 .5-.5zM8 5a.5.5 0 0 1 .5.5v7a.5.5 0 0 1-1 0v-7A.5.5 0 0 1 8 5zm3 .5v7a.5.5 0 0 1-1 0v-7a.5.5 0 0 1 1 0z" />
					</svg>
					Clear Points
				</button>
			</div>

			<div className="ignition-points-info">
				<span>Ignition Points: {ignitionPoints.length}</span>
				<p className="help-text">Click on the map to add ignition points</p>
			</div>
		</div>
	);
};

export default ControlPanel;
