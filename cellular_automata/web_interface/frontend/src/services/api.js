/**
 * API Service for Forest Fire Simulation
 * Provides methods for interacting with the Flask API endpoints
 */

import axios from "axios";

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || "/api";

// Create axios instance with common config
const apiClient = axios.create({
	baseURL: API_BASE_URL,
	headers: {
		"Content-Type": "application/json",
	},
});

// API service object with all available methods
const apiService = {
	/**
	 * Check API health status
	 * @returns {Promise} API health information
	 */
	async getHealth() {
		try {
			const response = await apiClient.get("/health");
			return response.data;
		} catch (error) {
			console.error("API Health check failed:", error);
			throw error;
		}
	},

	/**
	 * Get available simulation dates
	 * @returns {Promise} List of available dates for simulation
	 */
	async getAvailableDates() {
		try {
			const response = await apiClient.get("/available_dates");
			return response.data;
		} catch (error) {
			console.error("Failed to load available dates:", error);
			throw error;
		}
	},

	/**
	 * Run a fire simulation
	 * @param {Object} params - Simulation parameters
	 * @param {Array} params.ignition_points - Array of [x, y] or [lon, lat] coordinates
	 * @param {Object} params.weather_params - Weather parameters object
	 * @param {Number} params.simulation_hours - Number of hours to simulate
	 * @param {String} params.date - Simulation date in YYYY-MM-DD format
	 * @param {Boolean} params.use_ml_prediction - Whether to use ML model for prediction
	 * @returns {Promise} Simulation ID and initial status
	 */
	async runSimulation(params) {
		try {
			const response = await apiClient.post("/simulate", params);
			return response.data;
		} catch (error) {
			console.error("Simulation request failed:", error);
			throw error;
		}
	},

	/**
	 * Get status of a running simulation
	 * @param {String} simulationId - The unique ID of the simulation
	 * @returns {Promise} Current simulation status and results if completed
	 */
	async getSimulationStatus(simulationId) {
		try {
			const response = await apiClient.get(`/simulation/${simulationId}/status`);
			return response.data;
		} catch (error) {
			console.error(`Failed to get simulation ${simulationId} status:`, error);
			throw error;
		}
	},

	/**
	 * Get animation data for a completed simulation
	 * @param {String} simulationId - The unique ID of the simulation
	 * @returns {Promise} Animation frames and related data
	 */
	async getAnimationData(simulationId) {
		try {
			const response = await apiClient.get(`/simulation/${simulationId}/animation`);
			return response.data;
		} catch (error) {
			console.error(`Failed to get animation data for simulation ${simulationId}:`, error);
			throw error;
		}
	},

	/**
	 * Get API configuration
	 * @returns {Promise} API configuration parameters
	 */
	async getConfig() {
		try {
			const response = await apiClient.get("/config");
			return response.data;
		} catch (error) {
			console.error("Failed to load API configuration:", error);
			throw error;
		}
	},

	/**
	 * Run multiple scenarios for comparison
	 * @param {Object} params - Multiple scenario parameters
	 * @returns {Promise} Scenario comparison results
	 */
	async runMultipleScenarios(params) {
		try {
			const response = await apiClient.post("/multiple-scenarios", params);
			return response.data;
		} catch (error) {
			console.error("Multiple scenarios request failed:", error);
			throw error;
		}
	},

	/**
	 * Export simulation results as GeoTIFF
	 * @param {String} simulationId - The unique ID of the simulation
	 * @returns {Promise} Export URL
	 */
	async exportResults(simulationId) {
		try {
			const response = await apiClient.get(`/export-results/${simulationId}`);
			return response.data;
		} catch (error) {
			console.error(`Failed to export results for simulation ${simulationId}:`, error);
			throw error;
		}
	},
};

export default apiService;
