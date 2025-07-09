import React, { useEffect, useRef } from "react";
import { useSimulation } from "../context/SimulationContext";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";

const ResultsPanel = () => {
	const { state, setCurrentFrame, toggleAnimation } = useSimulation();

	const { currentResults, animationFrames, currentFrame, isAnimating, isLoading } = state;

	// If no results, show placeholder
	if (!currentResults && !isLoading) {
		return (
			<div className="card results-placeholder">
				<div className="card-header">
					<h2 className="card-title">
						<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="currentColor" viewBox="0 0 16 16">
							<path d="M8 16A8 8 0 1 0 8 0a8 8 0 0 0 0 16zm.25-11.25v3.25a.75.75 0 0 1-1.5 0V4.75a.75.75 0 0 1 1.5 0zm0 8a1 1 0 1 1-2 0 1 1 0 0 1 2 0z" />
						</svg>
						Simulation Results
					</h2>
				</div>
				<div className="results-empty-state">
					<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" fill="currentColor" viewBox="0 0 16 16">
						<path d="M6.95.435c.58-.58 1.52-.58 2.1 0l6.515 6.516c.58.58.58 1.519 0 2.098L9.05 15.565c-.58.58-1.519.58-2.098 0L.435 9.05a1.482 1.482 0 0 1 0-2.098L6.95.435zm1.4.7a.495.495 0 0 0-.7 0L1.134 7.65a.495.495 0 0 0 0 .7l6.516 6.516a.495.495 0 0 0 .7 0l6.516-6.516a.495.495 0 0 0 0-.7L8.35 1.134z" />
						<path d="M8 4a.5.5 0 0 1 .5.5v3h3a.5.5 0 0 1 0 1h-3v3a.5.5 0 0 1-1 0v-3h-3a.5.5 0 0 1 0-1h3v-3A.5.5 0 0 1 8 4z" />
					</svg>
					<p>No simulation results yet</p>
					<p className="hint">Set parameters and run a simulation to see results</p>
				</div>
			</div>
		);
	}

	// Return loading state if simulation is running
	if (isLoading) {
		return (
			<div className="card">
				<div className="card-header">
					<h2 className="card-title">
						<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="currentColor" viewBox="0 0 16 16">
							<path d="M8 3.5a.5.5 0 0 0-1 0V9a.5.5 0 0 0 .252.434l3.5 2a.5.5 0 0 0 .496-.868L8 8.71V3.5z" />
							<path d="M8 16A8 8 0 1 0 8 0a8 8 0 0 0 0 16zm7-8A7 7 0 1 1 1 8a7 7 0 0 1 14 0z" />
						</svg>
						Processing Simulation
					</h2>
				</div>
				<div className="loader">
					<div className="loader-spinner"></div>
					<p>Running fire spread simulation...</p>
				</div>
			</div>
		);
	}

	// Extract data for chart if available
	const hourlyData =
		currentResults?.hourly_statistics?.map((hour, index) => ({
			name: `Hour ${index + 1}`,
			burning: hour.total_burning_cells || 0,
			area: hour.burned_area_km2 || 0,
		})) || [];

	// Handle frame change
	const handleFrameChange = (e) => {
		setCurrentFrame(parseInt(e.target.value, 10));
	};

	// Handle animation toggle
	const handlePlayPause = () => {
		toggleAnimation();
	};

	// Animation timer for automatic frame advancement
	const animationTimer = useRef(null);

	// Handle animation playback
	useEffect(() => {
		// Clear any existing timer
		if (animationTimer.current) {
			clearInterval(animationTimer.current);
			animationTimer.current = null;
		}

		// Start new timer if animation is active
		if (isAnimating && animationFrames.length > 0) {
			animationTimer.current = setInterval(() => {
				setCurrentFrame((prevFrame) => {
					// Loop back to start when reaching the end
					if (prevFrame >= animationFrames.length - 1) {
						return 0;
					}
					return prevFrame + 1;
				});
			}, 1000); // 1 second per frame
		}

		// Cleanup on unmount
		return () => {
			if (animationTimer.current) {
				clearInterval(animationTimer.current);
			}
		};
	}, [isAnimating, animationFrames, setCurrentFrame]);

	return (
		<div className="card results-panel">
			<div className="card-header">
				<h2 className="card-title">
					<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="currentColor" viewBox="0 0 16 16">
						<path d="M8 16A8 8 0 1 0 8 0a8 8 0 0 0 0 16zm.25-11.25v3.25a.75.75 0 0 1-1.5 0V4.75a.75.75 0 0 1 1.5 0zm0 8a1 1 0 1 1-2 0 1 1 0 0 1 2 0z" />
					</svg>
					Simulation Results
				</h2>
				<span className="simulation-id">ID: {currentResults?.scenario_id}</span>
			</div>

			<div className="stat-grid">
				<div className="stat-card">
					<div className="stat-value">{currentResults?.total_hours_simulated || 0}</div>
					<div className="stat-label">Hours Simulated</div>
				</div>

				<div className="stat-card">
					<div className="stat-value">{currentResults?.final_statistics?.burned_area_km2?.toFixed(2) || 0}</div>
					<div className="stat-label">Total Area (km²)</div>
				</div>

				<div className="stat-card">
					<div className="stat-value">{currentResults?.final_statistics?.total_burning_cells || 0}</div>
					<div className="stat-label">Burning Cells</div>
				</div>

				<div className="stat-card">
					<div className="stat-value">{currentResults?.final_statistics?.fire_spread_rate?.toFixed(2) || 0}</div>
					<div className="stat-label">Spread Rate (m/min)</div>
				</div>
			</div>

			{animationFrames.length > 0 && (
				<div className="animation-controls">
					<button className="btn btn-icon" onClick={() => setCurrentFrame(Math.max(0, currentFrame - 1))} disabled={currentFrame === 0}>
						<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
							<path d="M11.354 1.646a.5.5 0 0 1 0 .708L5.707 8l5.647 5.646a.5.5 0 0 1-.708.708l-6-6a.5.5 0 0 1 0-.708l6-6a.5.5 0 0 1 .708 0z" />
						</svg>
					</button>

					<button className="btn btn-icon" onClick={handlePlayPause}>
						{isAnimating ? (
							<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
								<path d="M5.5 3.5A1.5 1.5 0 0 1 7 5v6a1.5 1.5 0 0 1-3 0V5a1.5 1.5 0 0 1 1.5-1.5zm5 0A1.5 1.5 0 0 1 12 5v6a1.5 1.5 0 0 1-3 0V5a1.5 1.5 0 0 1 1.5-1.5z" />
							</svg>
						) : (
							<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
								<path d="m11.596 8.697-6.363 3.692c-.54.313-1.233-.066-1.233-.697V4.308c0-.63.692-1.01 1.233-.696l6.363 3.692a.802.802 0 0 1 0 1.393z" />
							</svg>
						)}
					</button>

					<button className="btn btn-icon" onClick={() => setCurrentFrame(Math.min(animationFrames.length - 1, currentFrame + 1))} disabled={currentFrame === animationFrames.length - 1}>
						<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
							<path d="M4.646 1.646a.5.5 0 0 1 .708 0l6 6a.5.5 0 0 1 0 .708l-6 6a.5.5 0 0 1-.708-.708L10.293 8 4.646 2.354a.5.5 0 0 1 0-.708z" />
						</svg>
					</button>

					<span className="frame-counter">
						Frame {currentFrame + 1} of {animationFrames.length}
					</span>

					<input type="range" min="0" max={animationFrames.length - 1} value={currentFrame} onChange={handleFrameChange} className="animation-timeline" />
				</div>
			)}

			{hourlyData.length > 0 && (
				<div className="chart-container">
					<h3 className="chart-title">Fire Progression</h3>
					<ResponsiveContainer width="100%" height={300}>
						<BarChart data={hourlyData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
							<CartesianGrid strokeDasharray="3 3" />
							<XAxis dataKey="name" />
							<YAxis yAxisId="left" orientation="left" stroke="#FF6B35" />
							<YAxis yAxisId="right" orientation="right" stroke="#8884d8" />
							<Tooltip />
							<Legend />
							<Bar yAxisId="left" dataKey="burning" name="Burning Cells" fill="#FF6B35" />
							<Bar yAxisId="right" dataKey="area" name="Burned Area (km²)" fill="#8884d8" />
						</BarChart>
					</ResponsiveContainer>
				</div>
			)}

			<div className="export-section">
				<h3>Export Options</h3>
				<div className="export-buttons">
					<button
						className="btn btn-secondary"
						onClick={() => {
							if (currentResults?.scenario_id) {
								window.open(`/api/simulation/${currentResults.scenario_id}/export-csv`, "_blank");
							}
						}}
						disabled={!currentResults?.scenario_id}
					>
						<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
							<path d="M14 14V4.5L9.5 0H4a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2zM9.5 3A1.5 1.5 0 0 0 11 4.5h2V9H3V2a1 1 0 0 1 1-1h5.5v2zM3 12v-2h2v2H3zm0 1h2v2H4a1 1 0 0 1-1-1v-1zm3 2v-2h3v2H6zm4 0v-2h3v1a1 1 0 0 1-1 1h-2zm3-3h-3v-2h3v2zm-7 0v-2h3v2H6z" />
						</svg>
						CSV Report
					</button>
					<button
						className="btn btn-secondary"
						onClick={() => {
							if (currentResults?.scenario_id) {
								window.open(`/api/export-results/${currentResults.scenario_id}?format=geotiff`, "_blank");
							}
						}}
						disabled={!currentResults?.scenario_id}
					>
						<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
							<path d="M12.136.326A1.5 1.5 0 0 1 14 1.78V3h.5A1.5 1.5 0 0 1 16 4.5v9a1.5 1.5 0 0 1-1.5 1.5h-13A1.5 1.5 0 0 1 0 13.5v-9a1.5 1.5 0 0 1 1.432-1.499L12.136.326zM5.562 3H13V1.78a.5.5 0 0 0-.621-.484L5.562 3zM1.5 4a.5.5 0 0 0-.5.5v9a.5.5 0 0 0 .5.5h13a.5.5 0 0 0 .5-.5v-9a.5.5 0 0 0-.5-.5h-13z" />
						</svg>
						GeoTIFF
					</button>
					<button
						className="btn btn-secondary"
						onClick={() => {
							if (currentResults?.scenario_id) {
								window.open(`/api/simulation/${currentResults.scenario_id}/download-animation`, "_blank");
							}
						}}
						disabled={!currentResults?.scenario_id || animationFrames.length === 0}
					>
						<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
							<path d="M8.002 5.5a1.5 1.5 0 1 1-3 0 1.5 1.5 0 0 1 3 0z" />
							<path d="M12 0H4a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2V2a2 2 0 0 0-2-2zM3 2a1 1 0 0 1 1-1h8a1 1 0 0 1 1 1v8l-2.083-2.083a.5.5 0 0 0-.76.063L8 11 5.835 9.7a.5.5 0 0 0-.611.076L3 12V2z" />
						</svg>
						Animation
					</button>
				</div>

				<div className="export-info">
					<small>Export data for further analysis or visualization</small>
				</div>
			</div>
		</div>
	);
};

export default ResultsPanel;
