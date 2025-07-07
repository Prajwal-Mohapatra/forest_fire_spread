// UI Components for Forest Fire Simulation
// Handles interactive elements, modals, notifications, and UI state

class FireSimUI {
    constructor() {
        this.elements = {};
        this.state = {
            igniteMode: false,
            ignitionPoints: [],
            currentSimulation: null,
            animationState: 'stopped', // 'playing', 'paused', 'stopped'
            currentFrame: 0,
            maxFrames: 0,
            playbackSpeed: 1
        };
        this.eventListeners = new Map();
        this.init();
    }

    // Initialize UI components
    init() {
        this.cacheElements();
        this.setupEventListeners();
        this.initializeSliders();
        this.initializeToggles();
        this.loadDefaultValues();
        FireSimConfig.utils.log('info', 'UI components initialized');
    }

    // Cache DOM elements for performance
    cacheElements() {
        this.elements = {
            // Controls
            igniteToggle: document.getElementById('ignite-mode-toggle'),
            ignitionPointsList: document.getElementById('ignition-points-list'),
            clearIgnitionsBtn: document.getElementById('clear-ignitions-btn'),
            
            // Sliders
            ignitionIntensitySlider: document.getElementById('ignition-intensity-slider'),
            ignitionIntensityValue: document.getElementById('ignition-intensity-value'),
            simSpeedSlider: document.getElementById('sim-speed-slider'),
            simSpeedValue: document.getElementById('sim-speed-value'),
            timelineSlider: document.getElementById('timeline-slider'),
            timelineValue: document.getElementById('timeline-value'),
            
            // Inputs
            simulationDate: document.getElementById('simulation-date'),
            windSpeed: document.getElementById('wind-speed'),
            windDirection: document.getElementById('wind-direction'),
            temperature: document.getElementById('temperature'),
            humidity: document.getElementById('humidity'),
            simulationDuration: document.getElementById('simulation-duration'),
            resolution: document.getElementById('resolution'),
            physicsModel: document.getElementById('physics-model'),
            
            // Layer toggles
            layerProbability: document.getElementById('layer-probability'),
            layerFireSpread: document.getElementById('layer-fire-spread'),
            layerTerrain: document.getElementById('layer-terrain'),
            layerBarriers: document.getElementById('layer-barriers'),
            layerWeather: document.getElementById('layer-weather'),
            
            // Animation controls
            playBtn: document.getElementById('play-btn'),
            pauseBtn: document.getElementById('pause-btn'),
            stopBtn: document.getElementById('stop-btn'),
            frameBackBtn: document.getElementById('frame-back-btn'),
            frameForwardBtn: document.getElementById('frame-forward-btn'),
            speedBtns: document.querySelectorAll('.btn-speed'),
            
            // Main action
            startSimulationBtn: document.getElementById('start-simulation-btn'),
            exportResultsBtn: document.getElementById('export-results-btn'),
            
            // Display areas
            coordinatesDisplay: document.getElementById('coordinates-display'),
            progressSection: document.getElementById('progress-section'),
            progressFill: document.getElementById('progress-fill'),
            progressPercentage: document.getElementById('progress-percentage'),
            timelineSection: document.getElementById('timeline-section'),
            resultsSection: document.getElementById('results-section'),
            statisticsSection: document.getElementById('statistics-section'),
            
            // Timeline displays
            timelineStart: document.getElementById('timeline-start'),
            timelineIgnition: document.getElementById('timeline-ignition'),
            timelineRapid: document.getElementById('timeline-rapid'),
            timelineContainment: document.getElementById('timeline-containment'),
            timelineEnd: document.getElementById('timeline-end'),
            
            // Result displays
            resultArea: document.getElementById('result-area'),
            resultIntensity: document.getElementById('result-intensity'),
            resultTime: document.getElementById('result-time'),
            
            // Overlays and modals
            loadingOverlay: document.getElementById('loading-overlay'),
            loadingDetails: document.getElementById('loading-details'),
            errorModal: document.getElementById('error-modal'),
            errorMessage: document.getElementById('error-message'),
            successModal: document.getElementById('success-modal'),
            successMessage: document.getElementById('success-message')
        };
    }

    // Setup all event listeners
    setupEventListeners() {
        // Ignition mode toggle
        if (this.elements.igniteToggle) {
            this.addEventListener(this.elements.igniteToggle, 'click', () => {
                this.toggleIgniteMode();
            });
        }

        // Clear ignition points
        if (this.elements.clearIgnitionsBtn) {
            this.addEventListener(this.elements.clearIgnitionsBtn, 'click', () => {
                this.clearIgnitionPoints();
            });
        }

        // Animation controls
        if (this.elements.playBtn) {
            this.addEventListener(this.elements.playBtn, 'click', () => {
                this.playAnimation();
            });
        }

        if (this.elements.pauseBtn) {
            this.addEventListener(this.elements.pauseBtn, 'click', () => {
                this.pauseAnimation();
            });
        }

        if (this.elements.stopBtn) {
            this.addEventListener(this.elements.stopBtn, 'click', () => {
                this.stopAnimation();
            });
        }

        if (this.elements.frameBackBtn) {
            this.addEventListener(this.elements.frameBackBtn, 'click', () => {
                this.stepFrame(-1);
            });
        }

        if (this.elements.frameForwardBtn) {
            this.addEventListener(this.elements.frameForwardBtn, 'click', () => {
                this.stepFrame(1);
            });
        }

        // Speed buttons
        this.elements.speedBtns.forEach(btn => {
            this.addEventListener(btn, 'click', () => {
                const speed = parseFloat(btn.dataset.speed);
                this.setPlaybackSpeed(speed);
            });
        });

        // Timeline slider
        if (this.elements.timelineSlider) {
            this.addEventListener(this.elements.timelineSlider, 'input', (e) => {
                this.setCurrentFrame(parseInt(e.target.value));
            });
        }

        // Main simulation button
        if (this.elements.startSimulationBtn) {
            this.addEventListener(this.elements.startSimulationBtn, 'click', () => {
                window.FireSimManager.startSimulation();
            });
        }

        // Export button
        if (this.elements.exportResultsBtn) {
            this.addEventListener(this.elements.exportResultsBtn, 'click', () => {
                this.exportResults();
            });
        }

        // Modal close buttons
        document.querySelectorAll('.modal-close, #error-modal-ok, #success-modal-ok').forEach(btn => {
            this.addEventListener(btn, 'click', () => {
                this.hideAllModals();
            });
        });

        // Layer toggles
        Object.keys(FireSimConfig.layers).forEach(layerId => {
            const element = this.elements[`layer${layerId.charAt(0).toUpperCase() + layerId.slice(1)}`];
            if (element) {
                this.addEventListener(element, 'change', (e) => {
                    this.toggleLayer(layerId, e.target.checked);
                });
            }
        });

        FireSimConfig.utils.log('debug', 'Event listeners setup complete');
    }

    // Add event listener with cleanup tracking
    addEventListener(element, event, handler) {
        if (!element) return;
        
        element.addEventListener(event, handler);
        
        // Track for cleanup
        if (!this.eventListeners.has(element)) {
            this.eventListeners.set(element, []);
        }
        this.eventListeners.get(element).push({ event, handler });
    }

    // Initialize sliders with value synchronization
    initializeSliders() {
        const sliders = [
            { slider: this.elements.ignitionIntensitySlider, display: this.elements.ignitionIntensityValue },
            { slider: this.elements.simSpeedSlider, display: this.elements.simSpeedValue, suffix: 'x' },
            { slider: this.elements.timelineSlider, display: this.elements.timelineValue }
        ];

        sliders.forEach(({ slider, display, suffix = '' }) => {
            if (slider && display) {
                // Initialize display
                display.textContent = slider.value + suffix;
                
                // Sync on input
                this.addEventListener(slider, 'input', (e) => {
                    display.textContent = e.target.value + suffix;
                });
            }
        });
    }

    // Initialize toggle switches
    initializeToggles() {
        if (this.elements.igniteToggle) {
            this.elements.igniteToggle.classList.remove('active');
        }
    }

    // Load default values from configuration
    loadDefaultValues() {
        const defaults = FireSimConfig.simulation.defaultWeather;
        
        if (this.elements.windSpeed) this.elements.windSpeed.value = defaults.windSpeed;
        if (this.elements.windDirection) this.elements.windDirection.value = defaults.windDirection;
        if (this.elements.temperature) this.elements.temperature.value = defaults.temperature;
        if (this.elements.humidity) this.elements.humidity.value = defaults.humidity;
        
        if (this.elements.simulationDuration) {
            this.elements.simulationDuration.value = FireSimConfig.simulation.defaultDuration;
        }

        FireSimConfig.utils.log('debug', 'Default values loaded');
    }

    // Toggle ignite mode
    toggleIgniteMode() {
        this.state.igniteMode = !this.state.igniteMode;
        
        if (this.elements.igniteToggle) {
            this.elements.igniteToggle.classList.toggle('active', this.state.igniteMode);
        }

        // Update cursor on map
        const mapDisplay = document.getElementById('map-display');
        if (mapDisplay) {
            mapDisplay.style.cursor = this.state.igniteMode ? 'crosshair' : 'default';
        }

        FireSimConfig.utils.log('info', `Ignite mode ${this.state.igniteMode ? 'enabled' : 'disabled'}`);
    }

    // Add ignition point
    addIgnitionPoint(lat, lon) {
        if (this.state.ignitionPoints.length >= FireSimConfig.simulation.maxIgnitionPoints) {
            this.showError(`Maximum ${FireSimConfig.simulation.maxIgnitionPoints} ignition points allowed`);
            return false;
        }

        const point = { lat, lon, id: Date.now() };
        this.state.ignitionPoints.push(point);
        this.updateIgnitionPointsList();
        
        FireSimConfig.utils.log('info', 'Ignition point added:', point);
        return true;
    }

    // Remove ignition point
    removeIgnitionPoint(id) {
        this.state.ignitionPoints = this.state.ignitionPoints.filter(point => point.id !== id);
        this.updateIgnitionPointsList();
        
        FireSimConfig.utils.log('info', 'Ignition point removed:', id);
    }

    // Clear all ignition points
    clearIgnitionPoints() {
        this.state.ignitionPoints = [];
        this.updateIgnitionPointsList();
        
        // Clear from map
        if (window.FireSimMap) {
            window.FireSimMap.clearIgnitionPoints();
        }
        
        FireSimConfig.utils.log('info', 'All ignition points cleared');
    }

    // Update ignition points list display
    updateIgnitionPointsList() {
        if (!this.elements.ignitionPointsList) return;

        this.elements.ignitionPointsList.innerHTML = '';

        this.state.ignitionPoints.forEach((point, index) => {
            const item = document.createElement('div');
            item.className = 'ignition-point-item';
            item.innerHTML = `
                <div class="ignition-point-text">Point ${index + 1}</div>
                <button class="ignition-point-remove" data-id="${point.id}">×</button>
            `;

            const removeBtn = item.querySelector('.ignition-point-remove');
            this.addEventListener(removeBtn, 'click', () => {
                this.removeIgnitionPoint(point.id);
                if (window.FireSimMap) {
                    window.FireSimMap.removeIgnitionPoint(point.id);
                }
            });

            this.elements.ignitionPointsList.appendChild(item);
        });

        // Update start simulation button state
        if (this.elements.startSimulationBtn) {
            this.elements.startSimulationBtn.disabled = this.state.ignitionPoints.length === 0;
        }
    }

    // Toggle map layer
    toggleLayer(layerId, enabled) {
        if (window.FireSimMap) {
            window.FireSimMap.toggleLayer(layerId, enabled);
        }
        
        FireSimConfig.utils.log('info', `Layer ${layerId} ${enabled ? 'enabled' : 'disabled'}`);
    }

    // Animation controls
    playAnimation() {
        if (this.state.animationState === 'playing') return;
        
        this.state.animationState = 'playing';
        this.updateAnimationControls();
        
        if (window.FireSimManager) {
            window.FireSimManager.startAnimation();
        }
        
        FireSimConfig.utils.log('info', 'Animation started');
    }

    pauseAnimation() {
        if (this.state.animationState !== 'playing') return;
        
        this.state.animationState = 'paused';
        this.updateAnimationControls();
        
        if (window.FireSimManager) {
            window.FireSimManager.pauseAnimation();
        }
        
        FireSimConfig.utils.log('info', 'Animation paused');
    }

    stopAnimation() {
        this.state.animationState = 'stopped';
        this.state.currentFrame = 0;
        this.updateAnimationControls();
        this.updateTimelineDisplay();
        
        if (window.FireSimManager) {
            window.FireSimManager.stopAnimation();
        }
        
        FireSimConfig.utils.log('info', 'Animation stopped');
    }

    stepFrame(direction) {
        const newFrame = Math.max(0, Math.min(this.state.maxFrames - 1, 
                                              this.state.currentFrame + direction));
        this.setCurrentFrame(newFrame);
    }

    setCurrentFrame(frame) {
        this.state.currentFrame = frame;
        this.updateTimelineDisplay();
        
        if (window.FireSimManager) {
            window.FireSimManager.setFrame(frame);
        }
    }

    setPlaybackSpeed(speed) {
        this.state.playbackSpeed = speed;
        
        // Update button states
        this.elements.speedBtns.forEach(btn => {
            btn.classList.toggle('active', parseFloat(btn.dataset.speed) === speed);
        });
        
        if (window.FireSimManager) {
            window.FireSimManager.setPlaybackSpeed(speed);
        }
        
        FireSimConfig.utils.log('info', `Playback speed set to ${speed}x`);
    }

    // Update animation control states
    updateAnimationControls() {
        if (this.elements.playBtn) {
            this.elements.playBtn.disabled = this.state.animationState === 'playing';
        }
        if (this.elements.pauseBtn) {
            this.elements.pauseBtn.disabled = this.state.animationState !== 'playing';
        }
    }

    // Update timeline display
    updateTimelineDisplay() {
        if (this.elements.timelineSlider) {
            this.elements.timelineSlider.value = this.state.currentFrame;
            this.elements.timelineSlider.max = this.state.maxFrames - 1;
        }
        
        if (this.elements.timelineValue) {
            this.elements.timelineValue.textContent = this.state.currentFrame;
        }
    }

    // Set simulation results
    setSimulationResults(results) {
        this.state.currentSimulation = results;
        this.state.maxFrames = results.total_hours || 0;
        
        this.showResultsSection();
        this.updateResultsDisplay(results);
        this.updateTimelineSection(results);
        
        if (this.elements.timelineSlider) {
            this.elements.timelineSlider.max = Math.max(0, this.state.maxFrames - 1);
        }
    }

    // Show results section
    showResultsSection() {
        [this.elements.progressSection, this.elements.timelineSection, 
         this.elements.resultsSection, this.elements.statisticsSection].forEach(section => {
            if (section) section.style.display = 'block';
        });
    }

    // Update results display
    updateResultsDisplay(results) {
        const finalStats = results.hourly_statistics ? 
                          results.hourly_statistics[results.hourly_statistics.length - 1] : {};

        if (this.elements.resultArea) {
            const area = finalStats.burned_area_km2 || 0;
            this.elements.resultArea.textContent = `${(area * 100).toFixed(1)} hectares`;
        }

        if (this.elements.resultIntensity) {
            const intensity = finalStats.max_intensity || 0;
            this.elements.resultIntensity.textContent = `${intensity.toFixed(1)} kW/m`;
        }

        if (this.elements.resultTime) {
            this.elements.resultTime.textContent = `${results.total_hours || 0} hours`;
        }
    }

    // Update timeline section
    updateTimelineSection(results) {
        const duration = results.total_hours || 0;
        const milestones = {
            start: '00:00',
            ignition: '00:00',
            rapid: FireSimConfig.utils.formatTime(Math.floor(duration * 0.3 * 60)),
            containment: FireSimConfig.utils.formatTime(Math.floor(duration * 0.8 * 60)),
            end: FireSimConfig.utils.formatTime(duration * 60)
        };

        Object.keys(milestones).forEach(key => {
            const element = this.elements[`timeline${key.charAt(0).toUpperCase() + key.slice(1)}`];
            if (element) {
                element.textContent = milestones[key];
            }
        });
    }

    // Progress management
    showProgress(percentage = 0, details = 'Processing...') {
        if (this.elements.loadingOverlay) {
            this.elements.loadingOverlay.style.display = 'flex';
        }
        
        if (this.elements.loadingDetails) {
            this.elements.loadingDetails.textContent = details;
        }
        
        this.updateProgress(percentage);
    }

    updateProgress(percentage, details = null) {
        if (this.elements.progressFill) {
            this.elements.progressFill.style.width = `${percentage}%`;
        }
        
        if (this.elements.progressPercentage) {
            this.elements.progressPercentage.textContent = `${Math.round(percentage)}%`;
        }
        
        if (details && this.elements.loadingDetails) {
            this.elements.loadingDetails.textContent = details;
        }
    }

    hideProgress() {
        if (this.elements.loadingOverlay) {
            this.elements.loadingOverlay.style.display = 'none';
        }
    }

    // Modal management
    showError(message) {
        if (this.elements.errorMessage) {
            this.elements.errorMessage.textContent = message;
        }
        
        if (this.elements.errorModal) {
            this.elements.errorModal.style.display = 'flex';
        }
        
        FireSimConfig.utils.log('error', 'Error shown to user:', message);
    }

    showSuccess(message) {
        if (this.elements.successMessage) {
            this.elements.successMessage.textContent = message;
        }
        
        if (this.elements.successModal) {
            this.elements.successModal.style.display = 'flex';
        }
        
        FireSimConfig.utils.log('info', 'Success shown to user:', message);
    }

    hideAllModals() {
        [this.elements.errorModal, this.elements.successModal].forEach(modal => {
            if (modal) modal.style.display = 'none';
        });
    }

    // Update coordinates display
    updateCoordinates(lat, lon) {
        if (this.elements.coordinatesDisplay) {
            this.elements.coordinatesDisplay.textContent = 
                `Lat: ${lat.toFixed(4)}, Lon: ${lon.toFixed(4)}`;
        }
    }

    // Export results
    async exportResults() {
        if (!this.state.currentSimulation) {
            this.showError('No simulation results to export');
            return;
        }

        const simulationId = this.state.currentSimulation.simulation_id;
        if (!simulationId) {
            this.showError('Invalid simulation data');
            return;
        }

        try {
            this.showProgress(0, 'Preparing export...');
            
            const result = await window.FireSimAPI.exportResults(simulationId);
            
            if (result.success) {
                // Create download link
                const blob = new Blob([JSON.stringify(result.data, null, 2)], 
                                    { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `fire_simulation_${simulationId}_${new Date().toISOString().split('T')[0]}.json`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                
                this.hideProgress();
                this.showSuccess('Results exported successfully');
            } else {
                this.hideProgress();
                this.showError(`Export failed: ${result.error}`);
            }
        } catch (error) {
            this.hideProgress();
            this.showError(`Export error: ${error.message}`);
        }
    }

    // Get current form values
    getFormValues() {
        return {
            ignition_points: this.state.ignitionPoints.map(point => [point.lon, point.lat]),
            weather_params: {
                wind_direction: parseFloat(this.elements.windDirection?.value || 45),
                wind_speed: parseFloat(this.elements.windSpeed?.value || 15),
                temperature: parseFloat(this.elements.temperature?.value || 30),
                relative_humidity: parseFloat(this.elements.humidity?.value || 40)
            },
            simulation_hours: parseInt(this.elements.simulationDuration?.value || 6),
            date: this.elements.simulationDate?.value || '2016-05-15',
            use_ml_prediction: this.elements.physicsModel?.value === 'ml-enhanced',
            ignition_intensity: parseFloat(this.elements.ignitionIntensitySlider?.value || 50)
        };
    }

    // Cleanup
    destroy() {
        // Remove all event listeners
        this.eventListeners.forEach((handlers, element) => {
            handlers.forEach(({ event, handler }) => {
                element.removeEventListener(event, handler);
            });
        });
        this.eventListeners.clear();
        
        FireSimConfig.utils.log('info', 'UI components destroyed');
    }
}

// Create global UI instance
window.FireSimUI = new FireSimUI();

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = FireSimUI;
}
