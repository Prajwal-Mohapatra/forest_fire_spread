// API Service for Forest Fire Simulation
// Handles all communication with the backend API

class FireSimAPI {
    constructor(config) {
        this.config = config;
        this.baseUrl = config.api.baseUrl;
        this.timeout = config.api.timeout;
        this.retryAttempts = config.api.retryAttempts;
        this.retryDelay = config.api.retryDelay;
    }

    // Generic fetch wrapper with error handling and retries
    async fetchWithRetry(url, options = {}, attempt = 1) {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), this.timeout);

        const fetchOptions = {
            ...options,
            signal: controller.signal,
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            }
        };

        try {
            FireSimConfig.utils.log('debug', `API Request (attempt ${attempt}):`, { url, options: fetchOptions });
            
            const response = await fetch(url, fetchOptions);
            clearTimeout(timeoutId);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            FireSimConfig.utils.log('debug', 'API Response:', data);
            
            return { success: true, data };

        } catch (error) {
            clearTimeout(timeoutId);
            
            FireSimConfig.utils.log('error', `API Error (attempt ${attempt}):`, error.message);

            // Retry logic
            if (attempt < this.retryAttempts && !error.name === 'AbortError') {
                await this.delay(this.retryDelay * attempt);
                return this.fetchWithRetry(url, options, attempt + 1);
            }

            return { 
                success: false, 
                error: this.mapError(error),
                originalError: error
            };
        }
    }

    // Delay utility for retries
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    // Map errors to user-friendly messages
    mapError(error) {
        if (error.name === 'AbortError') {
            return FireSimConfig.errors.api.timeout;
        }
        if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
            return FireSimConfig.errors.api.connection;
        }
        if (error.message.includes('404')) {
            return FireSimConfig.errors.api.notFound;
        }
        if (error.message.includes('400')) {
            return FireSimConfig.errors.api.validation;
        }
        if (error.message.includes('500')) {
            return FireSimConfig.errors.api.server;
        }
        return error.message || 'Unknown error occurred';
    }

    // Get API endpoint URL
    getEndpointUrl(endpoint, params = {}) {
        return FireSimConfig.utils.getApiUrl(endpoint, params);
    }

    // Health check
    async checkHealth() {
        const url = this.getEndpointUrl('health');
        return this.fetchWithRetry(url);
    }

    // Get available dates
    async getAvailableDates() {
        const url = this.getEndpointUrl('dates');
        return this.fetchWithRetry(url);
    }

    // Get API configuration
    async getConfig() {
        const url = this.getEndpointUrl('config');
        return this.fetchWithRetry(url);
    }

    // Run simulation
    async runSimulation(parameters) {
        const url = this.getEndpointUrl('simulate');
        
        // Validate parameters
        const validation = this.validateSimulationParameters(parameters);
        if (!validation.valid) {
            return {
                success: false,
                error: validation.error
            };
        }

        const options = {
            method: 'POST',
            body: JSON.stringify(parameters)
        };

        return this.fetchWithRetry(url, options);
    }

    // Get simulation status
    async getSimulationStatus(simulationId) {
        if (!simulationId) {
            return { success: false, error: 'Simulation ID is required' };
        }

        const url = this.getEndpointUrl('status', { id: simulationId });
        return this.fetchWithRetry(url);
    }

    // Get animation data
    async getAnimationData(simulationId) {
        if (!simulationId) {
            return { success: false, error: 'Simulation ID is required' };
        }

        const url = this.getEndpointUrl('animation', { id: simulationId });
        return this.fetchWithRetry(url);
    }

    // Get simulation frame
    async getSimulationFrame(simulationId, hour) {
        if (!simulationId || hour === undefined) {
            return { success: false, error: 'Simulation ID and hour are required' };
        }

        const url = this.getEndpointUrl('frame', { id: simulationId, hour });
        
        // For binary data (images), we don't parse as JSON
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), this.timeout);

        try {
            const response = await fetch(url, { signal: controller.signal });
            clearTimeout(timeoutId);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const blob = await response.blob();
            return { success: true, data: blob };

        } catch (error) {
            clearTimeout(timeoutId);
            return { success: false, error: this.mapError(error) };
        }
    }

    // Run multiple scenarios
    async runMultipleScenarios(scenarios) {
        const url = this.getEndpointUrl('multipleScenarios');
        
        const options = {
            method: 'POST',
            body: JSON.stringify({ scenarios })
        };

        return this.fetchWithRetry(url, options);
    }

    // Get cached simulation
    async getCachedSimulation(simulationId) {
        if (!simulationId) {
            return { success: false, error: 'Simulation ID is required' };
        }

        const url = this.getEndpointUrl('cache', { id: simulationId });
        return this.fetchWithRetry(url);
    }

    // Export simulation results
    async exportResults(simulationId, format = 'json') {
        if (!simulationId) {
            return { success: false, error: 'Simulation ID is required' };
        }

        const url = this.getEndpointUrl('export', { id: simulationId });
        const options = {
            method: 'GET',
            headers: {
                'Accept': format === 'json' ? 'application/json' : 'application/octet-stream'
            }
        };

        return this.fetchWithRetry(url, options);
    }

    // Validate simulation parameters
    validateSimulationParameters(params) {
        const required = ['ignition_points', 'weather_params'];
        const missing = required.filter(field => !params[field]);
        
        if (missing.length > 0) {
            return {
                valid: false,
                error: `Missing required fields: ${missing.join(', ')}`
            };
        }

        // Validate ignition points
        if (!Array.isArray(params.ignition_points) || params.ignition_points.length === 0) {
            return {
                valid: false,
                error: FireSimConfig.errors.simulation.noIgnition
            };
        }

        // Validate coordinates
        for (const point of params.ignition_points) {
            if (!Array.isArray(point) || point.length !== 2) {
                return {
                    valid: false,
                    error: FireSimConfig.errors.simulation.invalidCoordinates
                };
            }
            
            const [lon, lat] = point;
            if (!FireSimConfig.utils.validateCoordinates(lat, lon)) {
                return {
                    valid: false,
                    error: FireSimConfig.errors.simulation.invalidCoordinates
                };
            }
        }

        // Validate weather parameters
        const weather = params.weather_params;
        const ranges = FireSimConfig.simulation.ranges;

        if (weather.wind_direction < ranges.windDirection.min || 
            weather.wind_direction > ranges.windDirection.max) {
            return {
                valid: false,
                error: FireSimConfig.errors.simulation.invalidWeather
            };
        }

        if (weather.wind_speed < ranges.windSpeed.min || 
            weather.wind_speed > ranges.windSpeed.max) {
            return {
                valid: false,
                error: FireSimConfig.errors.simulation.invalidWeather
            };
        }

        if (weather.temperature < ranges.temperature.min || 
            weather.temperature > ranges.temperature.max) {
            return {
                valid: false,
                error: FireSimConfig.errors.simulation.invalidWeather
            };
        }

        if (weather.relative_humidity < ranges.humidity.min || 
            weather.relative_humidity > ranges.humidity.max) {
            return {
                valid: false,
                error: FireSimConfig.errors.simulation.invalidWeather
            };
        }

        return { valid: true };
    }

    // Poll simulation status until completion
    async pollSimulationStatus(simulationId, onProgress = null, interval = 2000) {
        return new Promise((resolve, reject) => {
            const poll = async () => {
                try {
                    const result = await this.getSimulationStatus(simulationId);
                    
                    if (!result.success) {
                        reject(new Error(result.error));
                        return;
                    }

                    const status = result.data.status;
                    
                    if (onProgress) {
                        onProgress(result.data);
                    }

                    if (status === 'completed') {
                        resolve(result.data);
                    } else if (status === 'failed') {
                        reject(new Error('Simulation failed'));
                    } else {
                        setTimeout(poll, interval);
                    }
                } catch (error) {
                    reject(error);
                }
            };

            poll();
        });
    }

    // Batch request utility
    async batchRequest(requests) {
        const results = await Promise.allSettled(requests.map(req => 
            this.fetchWithRetry(req.url, req.options)
        ));

        return results.map((result, index) => ({
            request: requests[index],
            success: result.status === 'fulfilled' && result.value.success,
            data: result.status === 'fulfilled' ? result.value.data : null,
            error: result.status === 'rejected' ? result.reason.message : 
                   (result.value ? result.value.error : 'Unknown error')
        }));
    }
}

// Create global API instance
window.FireSimAPI = new FireSimAPI(window.FireSimConfig);

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = FireSimAPI;
}
