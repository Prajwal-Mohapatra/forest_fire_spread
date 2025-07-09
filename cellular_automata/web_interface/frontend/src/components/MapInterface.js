import React, { useEffect, useRef } from "react";
import { useSimulation } from "../context/SimulationContext";
import { MapContainer, TileLayer, Marker, Popup, LayersControl } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

// Fix for Leaflet marker icons in webpack environment
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
	iconRetinaUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
	iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
	shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
});

// Custom fire icon
const fireIcon = new L.Icon({
	iconUrl:
		"data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgZmlsbC1ydWxlPSJldmVub2RkIiBjbGlwLXJ1bGU9ImV2ZW5vZGQiPjxwYXRoIGZpbGw9IiNmZjZiMzUiIGQ9Ik04LjYyNSAyMWMuMjA5IDAgLjQxNi0uMDAyLjYyNS0uMDA3IDMuOTA1LS4wODggNi4yOTQtLjUwOCA5LjY4LTQuMDE0IDMuMDQzLTMuMTU2IDQuMTA4LTUuNTI4IDUuMDctOS44MzkuMzEzLjM0NC42NC42NzMgMS4wMjMuOTc5LTEuMDYxIDYuMjU2LTIuMDQyIDguMjE5LTUuMzQ0IDExLjY5NC0zLjM5NSAzLjU3NC03LjE4MSA0LjE4Ny0xMS45NTIgNC4xODctMi42OSAwLTQuNjY0LS40NDgtNS45OTEtMS4xNzEgMS45NjEtMS4wNDkgMy4wNDQtMS42MDYgNS4xODQtMS42MDYuODM2IDAgMS40OTEtLjA3NSAyLjI0NC0uMjI0aC4wMDF6bS0yLjE3Ni0xYy0yLjkzMSAwLTMuODExLjU2NS02Ljc4MSAxLjk2OSAyLjA1NS0zLjc4OCAzLjM3OS00LjcxNiA1Ljg0Ni03LjU2NC41MDMtLjU3OS45OCAtMS4wNjkgMS45NjgtMS45NiAxLjExOC0xLjAxMSAxLjkzOS0xLjYgMi42ODUtMi4xMTYgMS45MjktMS4zMzggMi45ODktMS45NzYgNC4zMzYtMy44MzcuNjU3LjM2NiAxLjQzNS43MS4yNjQgMS44ODIuMTczLjE3My4zOTQuMzQuNjQ1LjQ5NSAxLjQxMyA0Ljc5My43ODkgNy42MjctNS42OTQgMTEuMDE2LS42NzQuMTIzLTEuNDQyLjExNS0yLjI2OS4xMTV6bTEuMDU1LTkuOTc5Yy0yLjE0OS4xMi00LjUwNCAxLjY5OS01LjI0IDQuNjA5LTEuNDM3LS4yMzctMi40NC0uMTY3LTIuMzItMS4yMDEuMTMyLTEuMTMxLjQ5NS0yLjA1IDEuMDg4LTIuNzE5IDEuMjYzLTEuNDI0IDIuNTQ1LTEuMjIzIDMuNDQyLS44NTUtLjA0Ny0uNTY1LS4wNDMtLjkyMy4wMzQtMS42ODUgMi4yMTkgMS45MzEgMy41NzcgMS44NzcgNC45OTguMzY0IDEuMTkyIDEuNDQuNzE2IDMuNzA0LjcxNiA0LjEyMS0uMzYuNTE5LTEuNzE4LS4wNjgtMi43MTgtMi42MzR6Ii8+PC9zdmc+",
	iconSize: [32, 32],
	iconAnchor: [16, 32],
	popupAnchor: [0, -32],
});

const MapInterface = () => {
	const { state, addIgnitionPoint, removeIgnitionPoint } = useSimulation();
	const mapRef = useRef(null);

	const { mapCenter, mapZoom, baseLayer, ignitionPoints, currentResults, animationFrames, currentFrame } = state;

	// Handle map click to add ignition point
	const handleMapClick = (e) => {
		const { lat, lng } = e.latlng;
		addIgnitionPoint({ lat, lng });
	};

	// Handle remove ignition point
	const handleMarkerClick = (index) => {
		removeIgnitionPoint(index);
	};

	// Update map with simulation results if available
	useEffect(() => {
		if (!mapRef.current || !currentResults) return;

		const map = mapRef.current;

		// Clear previous overlays
		map.eachLayer((layer) => {
			if (layer._url && layer._url.includes("/api/simulation/")) {
				map.removeLayer(layer);
			}
		});

		// If we have animation frames and a valid current frame
		if (animationFrames?.length > 0 && currentFrame >= 0 && currentFrame < animationFrames.length) {
			// Add the current fire spread overlay
			const frameUrl = animationFrames[currentFrame];

			// Create ImageOverlay with the fire spread frame
			const bounds = map.getBounds();
			const overlay = L.imageOverlay(frameUrl, bounds, {
				opacity: 0.7,
				interactive: true,
			});

			overlay.addTo(map);

			// Add legend for the current frame
			if (!map._fireSpreadLegend) {
				const legend = L.control({ position: "bottomright" });

				legend.onAdd = function () {
					const div = L.DomUtil.create("div", "info legend");
					div.innerHTML = `
            <div class="legend-title">Fire Intensity</div>
            <div class="legend-item"><span style="background: #ffffb2"></span>Low</div>
            <div class="legend-item"><span style="background: #fed976"></span>Medium-Low</div>
            <div class="legend-item"><span style="background: #feb24c"></span>Medium</div>
            <div class="legend-item"><span style="background: #fd8d3c"></span>Medium-High</div>
            <div class="legend-item"><span style="background: #f03b20"></span>High</div>
            <div class="legend-item"><span style="background: #bd0026"></span>Very High</div>
          `;
					div.style.padding = "10px";
					div.style.backgroundColor = "white";
					div.style.borderRadius = "5px";
					div.style.boxShadow = "0 1px 5px rgba(0,0,0,0.4)";
					return div;
				};

				legend.addTo(map);
				map._fireSpreadLegend = legend;
			}
		}
	}, [currentResults, currentFrame, mapRef, animationFrames]);

	return (
		<div className="card map-container">
			<div className="card-header">
				<h2 className="card-title">
					<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="currentColor" viewBox="0 0 16 16">
						<path d="M8 16s6-5.686 6-10A6 6 0 0 0 2 6c0 4.314 6 10 6 10zm0-7a3 3 0 1 1 0-6 3 3 0 0 1 0 6z" />
					</svg>
					Map Interface
				</h2>
			</div>

			<div className="map-wrapper">
				<MapContainer
					center={mapCenter}
					zoom={mapZoom}
					style={{ height: "500px", width: "100%" }}
					whenCreated={(map) => {
						mapRef.current = map;
						map.on("click", handleMapClick);
					}}
				>
					<LayersControl position="topright">
						<LayersControl.BaseLayer name="OpenStreetMap" checked={baseLayer === "street"}>
							<TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors' />
						</LayersControl.BaseLayer>

						<LayersControl.BaseLayer name="Satellite" checked={baseLayer === "satellite"}>
							<TileLayer url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}" attribution='&copy; <a href="https://www.esri.com">Esri</a>' />
						</LayersControl.BaseLayer>

						<LayersControl.BaseLayer name="Terrain" checked={baseLayer === "terrain"}>
							<TileLayer url="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png" attribution='&copy; <a href="https://opentopomap.org">OpenTopoMap</a> contributors' />
						</LayersControl.BaseLayer>
					</LayersControl>

					{/* Display ignition points */}
					{ignitionPoints.map((point, index) => (
						<Marker
							key={index}
							position={[point.lat, point.lng]}
							icon={fireIcon}
							eventHandlers={{
								click: () => handleMarkerClick(index),
							}}
						>
							<Popup>
								<div>
									<strong>Ignition Point #{index + 1}</strong>
									<p>
										Lat: {point.lat.toFixed(4)}, Lng: {point.lng.toFixed(4)}
									</p>
									<button onClick={() => handleMarkerClick(index)}>Remove</button>
								</div>
							</Popup>
						</Marker>
					))}

					{/* Fire spread overlay would go here */}
					{/* This would be rendered conditionally when simulation results are available */}
				</MapContainer>
			</div>

			<div className="map-instructions">
				<p>Click on the map to add ignition points. Click on markers to remove them.</p>
			</div>
		</div>
	);
};

export default MapInterface;
