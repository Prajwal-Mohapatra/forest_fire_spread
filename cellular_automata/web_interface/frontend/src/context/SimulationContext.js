/**
 * SimulationContext for global state management
 * Provides state and actions for the entire simulation system
 */

import React, { createContext, useContext, useState, useReducer, useCallback } from "react";
import apiService from "../services/api";

// Initial state
const initialState = {
	// Simulation parameters
	availableDates: [],
	selectedDate: "",
	ignitionPoints: [],
	weatherParams: {
		wind_direction: 45,
		wind_speed: 15,
		temperature: 30,
		relative_humidity: 40,
	},
	simulationHours: 6,
	useMLPrediction: true,

	// System state
	isLoading: false,
	error: null,
	activeSimulation: null,
	completedSimulations: [],

	// Results
	currentResults: null,
	animationFrames: [],
	currentFrame: 0,
	isAnimating: false,

	// Map state
	mapCenter: [30.3165, 78.0322],
	mapZoom: 10,
	baseLayer: "satellite",
};

// Action types
const actionTypes = {
	SET_AVAILABLE_DATES: "SET_AVAILABLE_DATES",
	SELECT_DATE: "SELECT_DATE",
	ADD_IGNITION_POINT: "ADD_IGNITION_POINT",
	REMOVE_IGNITION_POINT: "REMOVE_IGNITION_POINT",
	CLEAR_IGNITION_POINTS: "CLEAR_IGNITION_POINTS",
	UPDATE_WEATHER_PARAM: "UPDATE_WEATHER_PARAM",
	SET_SIMULATION_HOURS: "SET_SIMULATION_HOURS",
	TOGGLE_ML_PREDICTION: "TOGGLE_ML_PREDICTION",

	SET_LOADING: "SET_LOADING",
	SET_ERROR: "SET_ERROR",
	CLEAR_ERROR: "CLEAR_ERROR",

	START_SIMULATION: "START_SIMULATION",
	SIMULATION_COMPLETED: "SIMULATION_COMPLETED",
	SIMULATION_FAILED: "SIMULATION_FAILED",

	SET_ANIMATION_FRAMES: "SET_ANIMATION_FRAMES",
	SET_CURRENT_FRAME: "SET_CURRENT_FRAME",
	TOGGLE_ANIMATION: "TOGGLE_ANIMATION",

	UPDATE_MAP_CENTER: "UPDATE_MAP_CENTER",
	UPDATE_MAP_ZOOM: "UPDATE_MAP_ZOOM",
	CHANGE_BASE_LAYER: "CHANGE_BASE_LAYER",
};

// Reducer function
function simulationReducer(state, action) {
	switch (action.type) {
		case actionTypes.SET_AVAILABLE_DATES:
			return { ...state, availableDates: action.payload };

		case actionTypes.SELECT_DATE:
			return { ...state, selectedDate: action.payload };

		case actionTypes.ADD_IGNITION_POINT:
			return {
				...state,
				ignitionPoints: [...state.ignitionPoints, action.payload],
			};

		case actionTypes.REMOVE_IGNITION_POINT:
			return {
				...state,
				ignitionPoints: state.ignitionPoints.filter((_, index) => index !== action.payload),
			};

		case actionTypes.CLEAR_IGNITION_POINTS:
			return { ...state, ignitionPoints: [] };

		case actionTypes.UPDATE_WEATHER_PARAM:
			return {
				...state,
				weatherParams: {
					...state.weatherParams,
					[action.payload.param]: action.payload.value,
				},
			};

		case actionTypes.SET_SIMULATION_HOURS:
			return { ...state, simulationHours: action.payload };

		case actionTypes.TOGGLE_ML_PREDICTION:
			return { ...state, useMLPrediction: !state.useMLPrediction };

		case actionTypes.SET_LOADING:
			return { ...state, isLoading: action.payload };

		case actionTypes.SET_ERROR:
			return { ...state, error: action.payload, isLoading: false };

		case actionTypes.CLEAR_ERROR:
			return { ...state, error: null };

		case actionTypes.START_SIMULATION:
			return {
				...state,
				activeSimulation: action.payload,
				isLoading: true,
				error: null,
			};

		case actionTypes.SIMULATION_COMPLETED:
			return {
				...state,
				activeSimulation: null,
				isLoading: false,
				completedSimulations: [...state.completedSimulations, action.payload.simulationId],
				currentResults: action.payload.results,
			};

		case actionTypes.SIMULATION_FAILED:
			return {
				...state,
				activeSimulation: null,
				isLoading: false,
				error: action.payload,
			};

		case actionTypes.SET_ANIMATION_FRAMES:
			return {
				...state,
				animationFrames: action.payload,
				currentFrame: 0,
			};

		case actionTypes.SET_CURRENT_FRAME:
			return { ...state, currentFrame: action.payload };

		case actionTypes.TOGGLE_ANIMATION:
			return { ...state, isAnimating: !state.isAnimating };

		case actionTypes.UPDATE_MAP_CENTER:
			return { ...state, mapCenter: action.payload };

		case actionTypes.UPDATE_MAP_ZOOM:
			return { ...state, mapZoom: action.payload };

		case actionTypes.CHANGE_BASE_LAYER:
			return { ...state, baseLayer: action.payload };

		default:
			return state;
	}
}

// Create context
const SimulationContext = createContext();

// Context provider component
export const SimulationProvider = ({ children }) => {
	const [state, dispatch] = useReducer(simulationReducer, initialState);

	// API actions
	const loadAvailableDates = useCallback(async () => {
		dispatch({ type: actionTypes.SET_LOADING, payload: true });
		try {
			const data = await apiService.getAvailableDates();
			dispatch({
				type: actionTypes.SET_AVAILABLE_DATES,
				payload: data.available_dates || [],
			});

			// Auto-select first date if available
			if (data.available_dates?.length > 0) {
				dispatch({
					type: actionTypes.SELECT_DATE,
					payload: data.available_dates[0].value,
				});
			}
		} catch (error) {
			dispatch({
				type: actionTypes.SET_ERROR,
				payload: "Failed to load available dates",
			});
		} finally {
			dispatch({ type: actionTypes.SET_LOADING, payload: false });
		}
	}, []);

	const runSimulation = useCallback(async () => {
		if (!state.selectedDate || state.ignitionPoints.length === 0) {
			dispatch({
				type: actionTypes.SET_ERROR,
				payload: "Please select a date and add ignition points",
			});
			return;
		}

		dispatch({ type: actionTypes.SET_LOADING, payload: true });

		try {
			const params = {
				ignition_points: state.ignitionPoints.map((p) => [p.x, p.y]),
				weather_params: state.weatherParams,
				simulation_hours: state.simulationHours,
				date: state.selectedDate,
				use_ml_prediction: state.useMLPrediction,
			};

			const result = await apiService.runSimulation(params);

			if (result.simulation_id) {
				dispatch({
					type: actionTypes.START_SIMULATION,
					payload: result.simulation_id,
				});

				// Poll for completion
				const pollStatus = async () => {
					const status = await apiService.getSimulationStatus(result.simulation_id);

					if (status.status === "completed") {
						const animationData = await apiService.getAnimationData(result.simulation_id);

						dispatch({
							type: actionTypes.SIMULATION_COMPLETED,
							payload: {
								simulationId: result.simulation_id,
								results: status.results || animationData,
							},
						});

						if (animationData.frame_urls) {
							dispatch({
								type: actionTypes.SET_ANIMATION_FRAMES,
								payload: animationData.frame_urls,
							});
						}
					} else if (status.status === "failed") {
						dispatch({
							type: actionTypes.SIMULATION_FAILED,
							payload: "Simulation failed: " + (status.error || "Unknown error"),
						});
					} else {
						setTimeout(pollStatus, 2000); // Poll every 2 seconds
					}
				};

				setTimeout(pollStatus, 1000);
			} else {
				dispatch({
					type: actionTypes.SIMULATION_FAILED,
					payload: "Invalid simulation response",
				});
			}
		} catch (error) {
			dispatch({
				type: actionTypes.SIMULATION_FAILED,
				payload: error.message || "Failed to run simulation",
			});
		}
	}, [state.selectedDate, state.ignitionPoints, state.weatherParams, state.simulationHours, state.useMLPrediction]);

	// Additional action creators
	const addIgnitionPoint = (point) => {
		dispatch({
			type: actionTypes.ADD_IGNITION_POINT,
			payload: point,
		});
	};

	const removeIgnitionPoint = (index) => {
		dispatch({
			type: actionTypes.REMOVE_IGNITION_POINT,
			payload: index,
		});
	};

	const clearIgnitionPoints = () => {
		dispatch({ type: actionTypes.CLEAR_IGNITION_POINTS });
	};

	const updateWeatherParam = (param, value) => {
		dispatch({
			type: actionTypes.UPDATE_WEATHER_PARAM,
			payload: { param, value: parseFloat(value) },
		});
	};

	const setSimulationHours = (hours) => {
		dispatch({
			type: actionTypes.SET_SIMULATION_HOURS,
			payload: parseInt(hours, 10),
		});
	};

	const toggleMLPrediction = () => {
		dispatch({ type: actionTypes.TOGGLE_ML_PREDICTION });
	};

	const setCurrentFrame = (frame) => {
		dispatch({
			type: actionTypes.SET_CURRENT_FRAME,
			payload: frame,
		});
	};

	const toggleAnimation = useCallback(() => {
		dispatch({ type: actionTypes.TOGGLE_ANIMATION });
	}, [dispatch]);

	const updateMapCenter = (center) => {
		dispatch({
			type: actionTypes.UPDATE_MAP_CENTER,
			payload: center,
		});
	};

	const updateMapZoom = (zoom) => {
		dispatch({
			type: actionTypes.UPDATE_MAP_ZOOM,
			payload: zoom,
		});
	};

	const changeBaseLayer = (layer) => {
		dispatch({
			type: actionTypes.CHANGE_BASE_LAYER,
			payload: layer,
		});
	};

	const clearError = () => {
		dispatch({ type: actionTypes.CLEAR_ERROR });
	};

	// Context value
	const value = {
		state,
		dispatch,
		loadAvailableDates,
		runSimulation,
		addIgnitionPoint,
		removeIgnitionPoint,
		clearIgnitionPoints,
		updateWeatherParam,
		setSimulationHours,
		toggleMLPrediction,
		setCurrentFrame,
		toggleAnimation,
		updateMapCenter,
		updateMapZoom,
		changeBaseLayer,
		clearError,
	};

	return <SimulationContext.Provider value={value}>{children}</SimulationContext.Provider>;
};

// Custom hook for using the context
export const useSimulation = () => {
	const context = useContext(SimulationContext);

	if (!context) {
		throw new Error("useSimulation must be used within a SimulationProvider");
	}

	return context;
};

export default SimulationContext;
