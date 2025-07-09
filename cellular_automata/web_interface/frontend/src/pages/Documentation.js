import React from "react";
import { Link } from "react-router-dom";
import "../styles/documentation.css";

const Documentation = () => {
	return (
		<div className="documentation-page">
			<header className="header">
				<div className="header-title">
					<div className="header-logo">
						<svg width="40" height="40" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
							<path d="M20 5C17.5 15 25 12.5 25 20C25 25 22.5 30 20 35C17.5 30 15 25 15 20C15 12.5 22.5 15 20 5Z" fill="#FF6B35" />
							<path d="M15 10C12.5 18 18 17 18 22.5C18 26 16 28.5 13.5 32.5C11 28.5 9 26 9 22.5C9 17 14.5 18 15 10Z" fill="#FF8F66" />
							<path d="M25 10C27.5 18 22 17 22 22.5C22 26 24 28.5 26.5 32.5C29 28.5 31 26 31 22.5C31 17 25.5 18 25 10Z" fill="#FF8F66" />
						</svg>
					</div>
					<div className="header-text">
						<h1>Forest Fire Spread Simulation</h1>
						<p>Documentation & Help</p>
					</div>
				</div>
				<div className="header-actions">
					<Link to="/" className="btn btn-primary">
						<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
							<path d="M8.707 1.5a1 1 0 0 0-1.414 0L.646 8.146a.5.5 0 0 0 .708.708L8 2.207l6.646 6.647a.5.5 0 0 0 .708-.708L13 5.793V2.5a.5.5 0 0 0-.5-.5h-1a.5.5 0 0 0-.5.5v1.293L8.707 1.5Z" />
							<path d="m8 3.293 6 6V13.5a1.5 1.5 0 0 1-1.5 1.5h-9A1.5 1.5 0 0 1 2 13.5V9.293l6-6Z" />
						</svg>
						<span>Back to Simulator</span>
					</Link>
				</div>
			</header>

			<main className="documentation-content">
				<div className="card">
					<h2>Getting Started</h2>
					<p>The Forest Fire Spread Simulation is an interactive tool for modeling and visualizing forest fire behavior. This guide will help you understand how to use the simulator.</p>
				</div>

				<div className="card">
					<h2>Using the Simulator</h2>

					<h3>Step 1: Select a Date</h3>
					<p>Choose a date from the dropdown menu in the control panel. Each date corresponds to a specific set of environmental conditions loaded from our database.</p>

					<h3>Step 2: Set Weather Parameters</h3>
					<p>Adjust the weather parameters using the sliders:</p>
					<ul>
						<li>
							<strong>Wind Direction</strong>: Direction the wind is blowing in degrees (0-360°)
						</li>
						<li>
							<strong>Wind Speed</strong>: Speed of the wind in km/h (0-50 km/h)
						</li>
						<li>
							<strong>Temperature</strong>: Ambient temperature in Celsius (10-50°C)
						</li>
						<li>
							<strong>Relative Humidity</strong>: Moisture content in the air (10-90%)
						</li>
					</ul>

					<h3>Step 3: Add Ignition Points</h3>
					<p>Click on the map to add one or more ignition points where the fire will start. You can remove points by clicking on them again.</p>

					<h3>Step 4: Run the Simulation</h3>
					<p>Click the "Run Simulation" button to start the fire spread simulation. The simulation will process based on your selected parameters and show the results when complete.</p>

					<h3>Step 5: View and Analyze Results</h3>
					<p>After the simulation completes, you can:</p>
					<ul>
						<li>View the animated progression of the fire</li>
						<li>See statistics on burned area and fire behavior</li>
						<li>Export the results in various formats</li>
					</ul>
				</div>

				<div className="card">
					<h2>Understanding the Map</h2>

					<h3>Base Layers</h3>
					<p>The map offers three base layer options:</p>
					<ul>
						<li>
							<strong>Street</strong>: OpenStreetMap view showing roads and settlements
						</li>
						<li>
							<strong>Satellite</strong>: Aerial imagery showing actual terrain
						</li>
						<li>
							<strong>Terrain</strong>: Topographic map showing elevation and landscape features
						</li>
					</ul>

					<h3>Map Symbols</h3>
					<ul>
						<li>
							🔥 <strong>Red Flame Icons</strong>: Ignition points
						</li>
						<li>
							🟥 <strong>Red Areas</strong>: Currently burning cells
						</li>
						<li>
							⬛ <strong>Black Areas</strong>: Burned areas
						</li>
						<li>
							🟩 <strong>Green Areas</strong>: Unburned vegetation
						</li>
					</ul>
				</div>

				<div className="card">
					<h2>Understanding Results</h2>

					<h3>Animation Controls</h3>
					<p>Use the playback controls to animate the fire spread over time:</p>
					<ul>
						<li>⏯️ Play/Pause the animation</li>
						<li>⏪ Move to previous frame</li>
						<li>⏩ Move to next frame</li>
						<li>🔄 Timeline slider to jump to any point in the simulation</li>
					</ul>

					<h3>Statistics</h3>
					<p>The results panel shows various statistics:</p>
					<ul>
						<li>
							<strong>Hours Simulated</strong>: Total duration of the fire simulation
						</li>
						<li>
							<strong>Total Area</strong>: Total burned area in square kilometers
						</li>
						<li>
							<strong>Burning Cells</strong>: Number of cells currently on fire
						</li>
						<li>
							<strong>Spread Rate</strong>: Average fire spread rate in meters per minute
						</li>
					</ul>

					<h3>Charts</h3>
					<p>The bar chart shows the hourly progression of:</p>
					<ul>
						<li>
							<strong>Burning Cells</strong>: Number of actively burning cells at each hour
						</li>
						<li>
							<strong>Burned Area</strong>: Cumulative area affected by fire at each hour
						</li>
					</ul>
				</div>

				<div className="card">
					<h2>Export Options</h2>
					<p>You can export simulation results in several formats:</p>
					<ul>
						<li>
							<strong>CSV Report</strong>: Tabular data with hourly statistics
						</li>
						<li>
							<strong>GeoTIFF</strong>: Geospatial raster data for use in GIS applications
						</li>
						<li>
							<strong>Animation</strong>: MP4 video showing the fire progression over time
						</li>
					</ul>
				</div>

				<div className="card">
					<h2>Technical Information</h2>
					<p>The simulation combines several advanced technologies:</p>
					<ul>
						<li>
							<strong>Machine Learning</strong>: Deep learning models predict fire spread based on environmental conditions
						</li>
						<li>
							<strong>Cellular Automata</strong>: Mathematical model simulates fire progression at a cellular level
						</li>
						<li>
							<strong>Remote Sensing</strong>: Satellite data provides accurate terrain and vegetation information
						</li>
					</ul>
				</div>
			</main>

			<footer className="documentation-footer">
				<p>Forest Fire Spread Simulation Project © 2025</p>
				<Link to="/">Return to Simulator</Link>
			</footer>
		</div>
	);
};

export default Documentation;
