// Map Handler for Forest Fire Simulation
// Manages Leaflet map, layers, markers, and geographic visualization

class FireSimMap {
    constructor() {
        this.map = null;
        this.layers = {
            base: {},
            overlay: {},
            markers: {}
        };
        this.ignitionMarkers = [];
        this.fireSpreadLayer = null;
        this.probabilityLayer = null;
        this.weatherOverlay = null;
        this.init();
    }

    // Initialize the map
    init() {
        try {
            this.createMap();
            this.setupBaseLayers();
            this.setupOverlayLayers();
            this.setupEventHandlers();
            this.setupControls();
            
            FireSimConfig.utils.log('info', 'Map initialized successfully');
        } catch (error) {
            FireSimConfig.utils.log('error', 'Map initialization failed:', error);
            this.showMapError('Failed to initialize map');
        }
    }

    // Create the main map instance
    createMap() {
        const config = FireSimConfig.map;
        const mapElement = document.getElementById('map-display');
        
        if (!mapElement) {
            throw new Error('Map container not found');
        }

        // Initialize Leaflet map
        this.map = L.map(mapElement, {
            center: [config.center.lat, config.center.lon],
            zoom: config.zoom.initial,
            minZoom: config.zoom.min,
            maxZoom: config.zoom.max,
            zoomControl: true,
            attributionControl: true
        });

        // Set bounds to Uttarakhand
        const bounds = L.latLngBounds(
            [config.bounds.south, config.bounds.west],
            [config.bounds.north, config.bounds.east]
        );
        this.map.setMaxBounds(bounds);
        this.map.fitBounds(bounds);
    }

    // Setup base map layers
    setupBaseLayers() {
        const layerConfigs = FireSimConfig.map.layers;

        // OpenStreetMap layer
        this.layers.base.street = L.tileLayer(layerConfigs.street.url, {
            attribution: layerConfigs.street.attribution,
            maxZoom: 18
        });

        // Satellite imagery layer
        this.layers.base.satellite = L.tileLayer(layerConfigs.satellite.url, {
            attribution: layerConfigs.satellite.attribution,
            maxZoom: 18
        });

        // Terrain layer
        this.layers.base.terrain = L.tileLayer(layerConfigs.terrain.url, {
            attribution: layerConfigs.terrain.attribution,
            subdomains: layerConfigs.terrain.subdomains,
            ext: layerConfigs.terrain.ext,
            maxZoom: 18
        });

        // Add default layer
        this.layers.base.street.addTo(this.map);
    }

    // Setup overlay layers for simulation data
    setupOverlayLayers() {
        // Fire probability layer group
        this.layers.overlay.probability = L.layerGroup();
        
        // Fire spread layer group
        this.layers.overlay.fireSpread = L.layerGroup().addTo(this.map);
        
        // Terrain overlay
        this.layers.overlay.terrain = L.layerGroup();
        
        // Barriers layer (roads, rivers, etc.)
        this.layers.overlay.barriers = L.layerGroup();
        
        // Weather data overlay
        this.layers.overlay.weather = L.layerGroup();
        
        // Ignition points layer
        this.layers.markers.ignition = L.layerGroup().addTo(this.map);
    }

    // Setup map event handlers
    setupEventHandlers() {
        // Click handler for adding ignition points
        this.map.on('click', (e) => {
            if (window.FireSimUI && window.FireSimUI.state.igniteMode) {
                this.addIgnitionPoint(e.latlng.lat, e.latlng.lng);
            }
        });

        // Mouse move handler for coordinate display
        this.map.on('mousemove', (e) => {
            if (window.FireSimUI) {
                window.FireSimUI.updateCoordinates(e.latlng.lat, e.latlng.lng);
            }
        });

        // Map bounds change handler
        this.map.on('moveend', () => {
            this.updateVisibleLayers();
        });

        // Zoom change handler
        this.map.on('zoomend', () => {
            this.updateMarkerSizes();
        });
    }

    // Setup map controls
    setupControls() {
        // Layer control
        const baseMaps = {
            "Street Map": this.layers.base.street,
            "Satellite": this.layers.base.satellite,
            "Terrain": this.layers.base.terrain
        };

        const overlayMaps = {
            "Fire Probability": this.layers.overlay.probability,
            "Fire Spread": this.layers.overlay.fireSpread,
            "Terrain Data": this.layers.overlay.terrain,
            "Barriers": this.layers.overlay.barriers,
            "Weather": this.layers.overlay.weather
        };

        L.control.layers(baseMaps, overlayMaps, {
            position: 'topright',
            collapsed: true
        }).addTo(this.map);

        // Scale control
        L.control.scale({
            position: 'bottomleft',
            metric: true,
            imperial: false
        }).addTo(this.map);

        // Add wind direction indicator
        this.addWindIndicator();
    }

    // Add ignition point marker
    addIgnitionPoint(lat, lon) {
        // Validate coordinates
        if (!FireSimConfig.utils.validateCoordinates(lat, lon)) {
            if (window.FireSimUI) {
                window.FireSimUI.showError(FireSimConfig.errors.map.outOfBounds);
            }
            return false;
        }

        // Add to UI state
        if (window.FireSimUI && !window.FireSimUI.addIgnitionPoint(lat, lon)) {
            return false;
        }

        // Create marker
        const markerId = Date.now();
        const marker = this.createIgnitionMarker(lat, lon, markerId);
        
        // Add to layer and track
        marker.addTo(this.layers.markers.ignition);
        this.ignitionMarkers.push({ id: markerId, marker, lat, lon });

        FireSimConfig.utils.log('info', 'Ignition point added to map:', { lat, lon, id: markerId });
        return true;
    }

    // Create ignition point marker
    createIgnitionMarker(lat, lon, id) {
        // Custom icon for ignition points
        const ignitionIcon = L.divIcon({
            className: 'ignition-marker',
            html: '',
            iconSize: [12, 12],
            iconAnchor: [6, 6]
        });

        const marker = L.marker([lat, lon], { icon: ignitionIcon });
        
        // Add popup
        marker.bindPopup(`
            <div style="font-family: ${FireSimConfig.ui.colors.primary};">
                <strong>Ignition Point</strong><br>
                Lat: ${lat.toFixed(4)}<br>
                Lon: ${lon.toFixed(4)}<br>
                <button onclick="window.FireSimMap.removeIgnitionPoint(${id})" 
                        style="margin-top: 8px; padding: 4px 8px; background: #F2730D; color: white; border: none; border-radius: 4px; cursor: pointer;">
                    Remove
                </button>
            </div>
        `);

        return marker;
    }

    // Remove ignition point
    removeIgnitionPoint(id) {
        const index = this.ignitionMarkers.findIndex(item => item.id === id);
        if (index === -1) return;

        const item = this.ignitionMarkers[index];
        this.layers.markers.ignition.removeLayer(item.marker);
        this.ignitionMarkers.splice(index, 1);

        // Update UI
        if (window.FireSimUI) {
            window.FireSimUI.removeIgnitionPoint(id);
        }

        FireSimConfig.utils.log('info', 'Ignition point removed from map:', id);
    }

    // Clear all ignition points
    clearIgnitionPoints() {
        this.layers.markers.ignition.clearLayers();
        this.ignitionMarkers = [];
        FireSimConfig.utils.log('info', 'All ignition points cleared from map');
    }

    // Toggle layer visibility
    toggleLayer(layerId, enabled) {
        const layerConfig = FireSimConfig.layers[layerId];
        if (!layerConfig) return;

        const layer = this.layers.overlay[layerId];
        if (!layer) return;

        if (enabled) {
            if (!this.map.hasLayer(layer)) {
                layer.addTo(this.map);
            }
        } else {
            if (this.map.hasLayer(layer)) {
                this.map.removeLayer(layer);
            }
        }

        FireSimConfig.utils.log('info', `Layer ${layerId} ${enabled ? 'enabled' : 'disabled'}`);
    }

    // Update fire probability layer
    updateProbabilityLayer(probabilityData) {
        // Clear existing layer
        this.layers.overlay.probability.clearLayers();

        if (!probabilityData || !probabilityData.features) return;

        // Create probability visualization
        const layer = L.geoJSON(probabilityData, {
            style: (feature) => {
                const probability = feature.properties.probability || 0;
                return {
                    fillColor: FireSimConfig.utils.getProbabilityColor(probability),
                    weight: 0,
                    opacity: 1,
                    color: 'transparent',
                    fillOpacity: FireSimConfig.layers.probability.opacity
                };
            },
            onEachFeature: (feature, layer) => {
                const prob = (feature.properties.probability * 100).toFixed(1);
                layer.bindPopup(`Fire Probability: ${prob}%`);
            }
        });

        layer.addTo(this.layers.overlay.probability);
        
        // Auto-enable if configured
        if (FireSimConfig.layers.probability.enabled) {
            this.layers.overlay.probability.addTo(this.map);
        }
    }

    // Update fire spread layer
    updateFireSpreadLayer(fireData, frameIndex = 0) {
        // Clear existing layer
        this.layers.overlay.fireSpread.clearLayers();

        if (!fireData || !fireData.frames || !fireData.frames[frameIndex]) return;

        const frameData = fireData.frames[frameIndex];
        
        // Create fire visualization
        const layer = L.geoJSON(frameData, {
            style: (feature) => {
                const intensity = feature.properties.intensity || 0;
                const state = feature.properties.state || 'unburned';
                
                let color;
                switch (state) {
                    case 'burning':
                        color = FireSimConfig.layers.fireSpread.colors.burning;
                        break;
                    case 'burned':
                        color = FireSimConfig.layers.fireSpread.colors.burned;
                        break;
                    default:
                        color = 'transparent';
                }

                return {
                    fillColor: color,
                    weight: 1,
                    opacity: intensity > 0 ? 1 : 0,
                    color: intensity > 0 ? '#ff4444' : 'transparent',
                    fillOpacity: FireSimConfig.layers.fireSpread.opacity
                };
            },
            onEachFeature: (feature, layer) => {
                const intensity = (feature.properties.intensity || 0).toFixed(3);
                const state = feature.properties.state || 'unburned';
                layer.bindPopup(`State: ${state}<br>Intensity: ${intensity}`);
            }
        });

        layer.addTo(this.layers.overlay.fireSpread);
    }

    // Add wind direction indicator
    addWindIndicator() {
        const windControl = L.control({ position: 'topright' });
        
        windControl.onAdd = function() {
            const div = L.DomUtil.create('div', 'wind-arrow');
            div.id = 'wind-indicator';
            div.title = 'Wind Direction';
            return div;
        };
        
        windControl.addTo(this.map);
        this.updateWindIndicator(45, 15); // Default values
    }

    // Update wind direction indicator
    updateWindIndicator(direction, speed) {
        const indicator = document.getElementById('wind-indicator');
        if (indicator) {
            // Rotate arrow to show wind direction
            const rotation = direction - 90; // Adjust for CSS rotation
            indicator.style.transform = `rotate(${rotation}deg)`;
            
            // Update tooltip
            indicator.title = `Wind: ${direction}° at ${speed} km/h`;
            
            // Add speed-based styling
            indicator.className = 'wind-arrow';
            if (speed > 30) {
                indicator.classList.add('strong-wind');
            } else if (speed > 15) {
                indicator.classList.add('moderate-wind');
            }
        }
    }

    // Update weather overlay
    updateWeatherOverlay(weatherData) {
        this.layers.overlay.weather.clearLayers();

        if (!weatherData) return;

        // Create weather visualization (simplified for demo)
        const weatherIcon = L.divIcon({
            className: 'weather-overlay',
            html: `
                <div style="background: rgba(255,255,255,0.9); padding: 8px; border-radius: 6px; font-size: 12px;">
                    🌡️ ${weatherData.temperature}°C<br>
                    💨 ${weatherData.wind_speed} km/h<br>
                    💧 ${weatherData.relative_humidity}%
                </div>
            `,
            iconSize: [100, 60],
            iconAnchor: [50, 30]
        });

        const weatherMarker = L.marker(
            [FireSimConfig.map.center.lat, FireSimConfig.map.center.lon], 
            { icon: weatherIcon }
        );

        weatherMarker.addTo(this.layers.overlay.weather);
    }

    // Update marker sizes based on zoom level
    updateMarkerSizes() {
        const zoom = this.map.getZoom();
        const scale = Math.max(0.5, Math.min(2, zoom / 10));
        
        // Update ignition marker sizes
        this.ignitionMarkers.forEach(item => {
            const element = item.marker.getElement();
            if (element) {
                element.style.transform = `scale(${scale})`;
            }
        });
    }

    // Update visible layers based on zoom and bounds
    updateVisibleLayers() {
        const zoom = this.map.getZoom();
        const bounds = this.map.getBounds();
        
        // Performance optimization: hide complex layers at low zoom
        if (zoom < 8) {
            // Hide detailed layers for performance
            if (this.map.hasLayer(this.layers.overlay.barriers)) {
                this.map.removeLayer(this.layers.overlay.barriers);
            }
        } else {
            // Show detailed layers at higher zoom
            const barriersEnabled = document.getElementById('layer-barriers')?.checked;
            if (barriersEnabled && !this.map.hasLayer(this.layers.overlay.barriers)) {
                this.layers.overlay.barriers.addTo(this.map);
            }
        }
    }

    // Show map error
    showMapError(message) {
        const mapElement = document.getElementById('map-display');
        if (mapElement) {
            mapElement.innerHTML = `
                <div style="
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    height: 100%;
                    background: var(--background-section);
                    color: var(--primary-text);
                    text-align: center;
                    padding: 20px;
                ">
                    <div style="font-size: 48px; margin-bottom: 16px;">🗺️</div>
                    <div style="font-size: 18px; font-weight: 600; margin-bottom: 8px;">Map Error</div>
                    <div style="font-size: 14px; opacity: 0.8;">${message}</div>
                    <button onclick="window.location.reload()" style="
                        margin-top: 16px;
                        padding: 8px 16px;
                        background: var(--accent-orange);
                        color: white;
                        border: none;
                        border-radius: 6px;
                        cursor: pointer;
                    ">Reload Page</button>
                </div>
            `;
        }
    }

    // Get current map view
    getMapView() {
        if (!this.map) return null;
        
        return {
            center: this.map.getCenter(),
            zoom: this.map.getZoom(),
            bounds: this.map.getBounds()
        };
    }

    // Set map view
    setMapView(center, zoom) {
        if (this.map) {
            this.map.setView(center, zoom);
        }
    }

    // Fit map to simulation bounds
    fitToSimulationBounds() {
        const bounds = L.latLngBounds(
            [FireSimConfig.map.bounds.south, FireSimConfig.map.bounds.west],
            [FireSimConfig.map.bounds.north, FireSimConfig.map.bounds.east]
        );
        
        if (this.map) {
            this.map.fitBounds(bounds, { padding: [20, 20] });
        }
    }

    // Export map as image
    async exportMapImage() {
        if (!this.map) return null;
        
        try {
            // Use html2canvas or similar library to capture map
            // This is a simplified implementation
            const mapElement = document.getElementById('map-display');
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            // Set canvas size
            canvas.width = mapElement.offsetWidth;
            canvas.height = mapElement.offsetHeight;
            
            // Draw background color
            ctx.fillStyle = FireSimConfig.ui.colors.background;
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            // Add "Map Export" text (placeholder)
            ctx.fillStyle = FireSimConfig.ui.colors.primary;
            ctx.font = '16px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('Map Export', canvas.width / 2, canvas.height / 2);
            
            return canvas.toDataURL('image/png');
        } catch (error) {
            FireSimConfig.utils.log('error', 'Map export failed:', error);
            return null;
        }
    }

    // Cleanup
    destroy() {
        if (this.map) {
            this.map.remove();
            this.map = null;
        }
        
        this.layers = { base: {}, overlay: {}, markers: {} };
        this.ignitionMarkers = [];
        
        FireSimConfig.utils.log('info', 'Map destroyed');
    }
}

// Create global map instance
window.FireSimMap = new FireSimMap();

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = FireSimMap;
}
