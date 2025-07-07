// Simulation Manager for Forest Fire Simulation
// Orchestrates simulation runs, polling, animation, and data flow

class FireSimulationManager {
    constructor() {
        this.currentSimulation = null;
        this.animationTimer = null;
        this.simulationData = null;
        this.animationFrames = [];
        this.animationState = 'stopped'; // 'stopped', 'playing', 'paused'
        this.currentFrame = 0;
        this.playbackSpeed = 1;
        this.pollingInterval = null;
        this.frameUpdateCallback = null;
        
        this.init();
    }

    // Initialize simulation manager
    init() {
        this.bindEventHandlers();
        FireSimConfig.utils.log('info', 'Simulation manager initialized');
    }

    // Bind event handlers
    bindEventHandlers() {
        // Listen for UI events if needed
        document.addEventListener('visibilitychange', () => {
            if (document.hidden && this.animationState === 'playing') {
                this.pauseAnimation();
            }
        });
    }

    // Start a new simulation
    async startSimulation() {
        try {
            // Validate UI state
            if (!window.FireSimUI) {
                throw new Error('UI not initialized');
            }

            // Get form values
            const parameters = window.FireSimUI.getFormValues();
            
            // Validate ignition points
            if (parameters.ignition_points.length === 0) {
                window.FireSimUI.showError(FireSimConfig.errors.simulation.noIgnition);
                return;
            }

            // Show progress
            window.FireSimUI.showProgress(0, 'Starting simulation...');

            // Send simulation request
            const result = await window.FireSimAPI.runSimulation(parameters);

            if (!result.success) {
                window.FireSimUI.hideProgress();
                window.FireSimUI.showError(`Simulation failed: ${result.error}`);
                return;
            }

            // Store simulation reference
            this.currentSimulation = result.data;
            const simulationId = result.data.simulation_id;

            FireSimConfig.utils.log('info', 'Simulation started:', simulationId);

            // Update progress
            window.FireSimUI.updateProgress(10, 'Simulation running...');

            // Start polling for status
            this.startStatusPolling(simulationId);

        } catch (error) {
            window.FireSimUI.hideProgress();
            window.FireSimUI.showError(`Error starting simulation: ${error.message}`);
            FireSimConfig.utils.log('error', 'Simulation start error:', error);
        }
    }

    // Start polling simulation status
    startStatusPolling(simulationId) {
        let progressCounter = 10;

        const poll = async () => {
            try {
                const result = await window.FireSimAPI.getSimulationStatus(simulationId);
                
                if (!result.success) {
                    this.stopStatusPolling();
                    window.FireSimUI.hideProgress();
                    window.FireSimUI.showError(`Status check failed: ${result.error}`);
                    return;
                }

                const statusData = result.data;
                const status = statusData.status;
                
                FireSimConfig.utils.log('debug', 'Simulation status:', statusData);

                // Update progress
                progressCounter = Math.min(90, progressCounter + 5);
                const progressMessage = this.getProgressMessage(status, statusData);
                window.FireSimUI.updateProgress(progressCounter, progressMessage);

                // Handle status
                switch (status) {
                    case 'completed':
                        this.stopStatusPolling();
                        await this.handleSimulationComplete(simulationId, statusData);
                        break;

                    case 'failed':
                        this.stopStatusPolling();
                        window.FireSimUI.hideProgress();
                        window.FireSimUI.showError(
                            statusData.error || FireSimConfig.errors.simulation.failed
                        );
                        break;

                    case 'running':
                        // Continue polling
                        break;

                    default:
                        FireSimConfig.utils.log('warn', 'Unknown simulation status:', status);
                }

            } catch (error) {
                this.stopStatusPolling();
                window.FireSimUI.hideProgress();
                window.FireSimUI.showError(`Polling error: ${error.message}`);
                FireSimConfig.utils.log('error', 'Status polling error:', error);
            }
        };

        // Start polling
        this.pollingInterval = setInterval(poll, 2000);
        poll(); // Initial poll
    }

    // Stop status polling
    stopStatusPolling() {
        if (this.pollingInterval) {
            clearInterval(this.pollingInterval);
            this.pollingInterval = null;
        }
    }

    // Get progress message based on status
    getProgressMessage(status, statusData) {
        switch (status) {
            case 'initializing':
                return 'Initializing simulation engine...';
            case 'loading_data':
                return 'Loading environmental data...';
            case 'running':
                const progress = statusData.progress || 0;
                const step = statusData.current_step || 0;
                const totalSteps = statusData.total_steps || 100;
                return `Running simulation... Step ${step}/${totalSteps} (${Math.round(progress)}%)`;
            case 'processing_results':
                return 'Processing simulation results...';
            case 'completed':
                return 'Simulation completed successfully!';
            default:
                return 'Processing...';
        }
    }

    // Handle simulation completion
    async handleSimulationComplete(simulationId, statusData) {
        try {
            // Update progress
            window.FireSimUI.updateProgress(90, 'Loading animation data...');

            // Get animation data
            const animationResult = await window.FireSimAPI.getAnimationData(simulationId);
            
            if (!animationResult.success) {
                window.FireSimUI.hideProgress();
                window.FireSimUI.showError(`Failed to load animation: ${animationResult.error}`);
                return;
            }

            // Store simulation data
            this.simulationData = animationResult.data;
            this.animationFrames = this.simulationData.frames || [];

            // Update progress
            window.FireSimUI.updateProgress(95, 'Preparing visualization...');

            // Set simulation results in UI
            window.FireSimUI.setSimulationResults({
                simulation_id: simulationId,
                total_hours: this.simulationData.total_hours || this.animationFrames.length,
                hourly_statistics: this.simulationData.hourly_statistics || [],
                final_statistics: this.simulationData.final_statistics || {}
            });

            // Update charts
            if (window.FireSimCharts) {
                await window.FireSimCharts.updateChartsFromSimulation(this.simulationData);
            }

            // Load initial frame
            if (this.animationFrames.length > 0) {
                await this.loadFrame(0);
            }

            // Final progress update
            window.FireSimUI.updateProgress(100, 'Complete!');
            
            // Hide progress after a short delay
            setTimeout(() => {
                window.FireSimUI.hideProgress();
                window.FireSimUI.showSuccess(FireSimConfig.messages.simulation.completed);
            }, 1000);

            FireSimConfig.utils.log('info', 'Simulation completed successfully');

        } catch (error) {
            window.FireSimUI.hideProgress();
            window.FireSimUI.showError(`Error processing results: ${error.message}`);
            FireSimConfig.utils.log('error', 'Simulation completion error:', error);
        }
    }

    // Load a specific animation frame
    async loadFrame(frameIndex) {
        if (!this.animationFrames || frameIndex >= this.animationFrames.length) {
            return;
        }

        try {
            this.currentFrame = frameIndex;
            const frameData = this.animationFrames[frameIndex];

            // Update map layers
            if (window.FireSimMap) {
                // Update fire spread layer
                window.FireSimMap.updateFireSpreadLayer(
                    { frames: this.animationFrames }, 
                    frameIndex
                );

                // Update probability layer if available
                if (frameData.probability_data) {
                    window.FireSimMap.updateProbabilityLayer(frameData.probability_data);
                }
            }

            // Update UI timeline
            if (window.FireSimUI) {
                window.FireSimUI.state.currentFrame = frameIndex;
                window.FireSimUI.updateTimelineDisplay();
            }

            // Update charts for current frame
            if (window.FireSimCharts && this.simulationData.hourly_statistics) {
                const currentStats = this.simulationData.hourly_statistics[frameIndex];
                if (currentStats) {
                    window.FireSimCharts.updateCurrentFrameData(currentStats, frameIndex);
                }
            }

            // Call frame update callback if set
            if (this.frameUpdateCallback) {
                this.frameUpdateCallback(frameIndex, frameData);
            }

            FireSimConfig.utils.log('debug', `Frame ${frameIndex} loaded`);

        } catch (error) {
            FireSimConfig.utils.log('error', 'Frame loading error:', error);
        }
    }

    // Animation controls
    startAnimation() {
        if (this.animationState === 'playing') return;
        if (!this.animationFrames || this.animationFrames.length === 0) return;

        this.animationState = 'playing';
        
        const frameDelay = FireSimConfig.simulation.animation.speedMultipliers[`${this.playbackSpeed}x`] || 1000;
        
        this.animationTimer = setInterval(() => {
            if (this.currentFrame >= this.animationFrames.length - 1) {
                // Animation complete
                this.stopAnimation();
                return;
            }

            this.loadFrame(this.currentFrame + 1);
        }, frameDelay);

        FireSimConfig.utils.log('info', `Animation started at ${this.playbackSpeed}x speed`);
    }

    pauseAnimation() {
        if (this.animationState !== 'playing') return;

        this.animationState = 'paused';
        
        if (this.animationTimer) {
            clearInterval(this.animationTimer);
            this.animationTimer = null;
        }

        FireSimConfig.utils.log('info', 'Animation paused');
    }

    stopAnimation() {
        this.animationState = 'stopped';
        
        if (this.animationTimer) {
            clearInterval(this.animationTimer);
            this.animationTimer = null;
        }

        // Reset to first frame
        if (this.animationFrames.length > 0) {
            this.loadFrame(0);
        }

        FireSimConfig.utils.log('info', 'Animation stopped');
    }

    setFrame(frameIndex) {
        if (frameIndex < 0 || frameIndex >= this.animationFrames.length) return;
        
        // Pause animation if playing
        if (this.animationState === 'playing') {
            this.pauseAnimation();
        }

        this.loadFrame(frameIndex);
    }

    setPlaybackSpeed(speed) {
        this.playbackSpeed = speed;

        // Restart animation with new speed if currently playing
        if (this.animationState === 'playing') {
            this.pauseAnimation();
            this.startAnimation();
        }

        FireSimConfig.utils.log('info', `Playback speed set to ${speed}x`);
    }

    // Run multiple scenario comparison
    async runMultipleScenarios(scenarios) {
        try {
            window.FireSimUI.showProgress(0, 'Starting multiple scenarios...');

            const result = await window.FireSimAPI.runMultipleScenarios(scenarios);

            if (!result.success) {
                window.FireSimUI.hideProgress();
                window.FireSimUI.showError(`Multiple scenarios failed: ${result.error}`);
                return;
            }

            // Handle multiple scenario results
            const scenarioResults = result.data.scenarios || [];
            
            // Update UI with comparison view
            this.displayScenarioComparison(scenarioResults);

            window.FireSimUI.hideProgress();
            window.FireSimUI.showSuccess('Multiple scenarios completed');

            FireSimConfig.utils.log('info', 'Multiple scenarios completed:', scenarioResults.length);

        } catch (error) {
            window.FireSimUI.hideProgress();
            window.FireSimUI.showError(`Error running multiple scenarios: ${error.message}`);
            FireSimConfig.utils.log('error', 'Multiple scenarios error:', error);
        }
    }

    // Display scenario comparison
    displayScenarioComparison(scenarios) {
        // This would typically create a comparison view
        // For now, we'll just log the results and show basic comparison
        
        if (window.FireSimCharts) {
            window.FireSimCharts.createScenarioComparison(scenarios);
        }

        // Show comparison results in UI
        const comparisonData = scenarios.map((scenario, index) => ({
            name: `Scenario ${index + 1}`,
            burned_area: scenario.final_statistics?.burned_area_km2 || 0,
            max_intensity: scenario.final_statistics?.max_intensity || 0,
            duration: scenario.total_hours || 0
        }));

        FireSimConfig.utils.log('info', 'Scenario comparison data:', comparisonData);
    }

    // Run demo scenario
    async runDemoScenario(scenarioName) {
        const demo = FireSimConfig.demos[scenarioName];
        if (!demo) {
            window.FireSimUI.showError('Demo scenario not found');
            return;
        }

        try {
            // Clear existing ignition points
            if (window.FireSimUI) {
                window.FireSimUI.clearIgnitionPoints();
            }
            if (window.FireSimMap) {
                window.FireSimMap.clearIgnitionPoints();
            }

            // Set demo parameters
            const params = demo.parameters;
            
            // Add ignition points
            params.ignitionPoints.forEach(point => {
                if (window.FireSimMap) {
                    window.FireSimMap.addIgnitionPoint(point.lat, point.lon);
                }
            });

            // Set weather parameters in UI
            if (window.FireSimUI.elements.windSpeed) {
                window.FireSimUI.elements.windSpeed.value = params.weather.windSpeed;
            }
            if (window.FireSimUI.elements.windDirection) {
                window.FireSimUI.elements.windDirection.value = params.weather.windDirection;
            }
            if (window.FireSimUI.elements.temperature) {
                window.FireSimUI.elements.temperature.value = params.weather.temperature;
            }
            if (window.FireSimUI.elements.humidity) {
                window.FireSimUI.elements.humidity.value = params.weather.humidity;
            }
            if (window.FireSimUI.elements.simulationDuration) {
                window.FireSimUI.elements.simulationDuration.value = params.duration;
            }
            if (window.FireSimUI.elements.simulationDate) {
                window.FireSimUI.elements.simulationDate.value = params.date;
            }

            // Update wind indicator
            if (window.FireSimMap) {
                window.FireSimMap.updateWindIndicator(
                    params.weather.windDirection, 
                    params.weather.windSpeed
                );
            }

            window.FireSimUI.showSuccess(`${demo.name} loaded successfully`);
            
            // Auto-start simulation after a short delay
            setTimeout(() => {
                this.startSimulation();
            }, 1500);

            FireSimConfig.utils.log('info', 'Demo scenario loaded:', scenarioName);

        } catch (error) {
            window.FireSimUI.showError(`Error loading demo: ${error.message}`);
            FireSimConfig.utils.log('error', 'Demo loading error:', error);
        }
    }

    // Get current simulation state
    getSimulationState() {
        return {
            currentSimulation: this.currentSimulation,
            simulationData: this.simulationData,
            animationState: this.animationState,
            currentFrame: this.currentFrame,
            totalFrames: this.animationFrames.length,
            playbackSpeed: this.playbackSpeed
        };
    }

    // Set frame update callback
    setFrameUpdateCallback(callback) {
        this.frameUpdateCallback = callback;
    }

    // Cache simulation data for quick access
    cacheSimulationData(simulationId) {
        if (this.simulationData) {
            try {
                localStorage.setItem(`simulation_${simulationId}`, JSON.stringify({
                    data: this.simulationData,
                    timestamp: Date.now()
                }));
                FireSimConfig.utils.log('info', 'Simulation data cached');
            } catch (error) {
                FireSimConfig.utils.log('warn', 'Failed to cache simulation data:', error);
            }
        }
    }

    // Load cached simulation data
    loadCachedSimulation(simulationId) {
        try {
            const cached = localStorage.getItem(`simulation_${simulationId}`);
            if (cached) {
                const cachedData = JSON.parse(cached);
                
                // Check if cache is still valid (24 hours)
                const cacheAge = Date.now() - cachedData.timestamp;
                if (cacheAge < 24 * 60 * 60 * 1000) {
                    this.simulationData = cachedData.data;
                    this.animationFrames = this.simulationData.frames || [];
                    FireSimConfig.utils.log('info', 'Simulation data loaded from cache');
                    return true;
                }
            }
        } catch (error) {
            FireSimConfig.utils.log('warn', 'Failed to load cached simulation:', error);
        }
        return false;
    }

    // Cleanup
    destroy() {
        this.stopAnimation();
        this.stopStatusPolling();
        
        this.currentSimulation = null;
        this.simulationData = null;
        this.animationFrames = [];
        
        FireSimConfig.utils.log('info', 'Simulation manager destroyed');
    }
}

// Create global simulation manager instance
window.FireSimManager = new FireSimulationManager();

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = FireSimulationManager;
}
