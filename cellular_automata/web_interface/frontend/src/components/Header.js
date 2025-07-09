import React from "react";
import { Link, useNavigate } from "react-router-dom";
import { useSimulation } from "../context/SimulationContext";

const Header = () => {
	const navigate = useNavigate();
	const { state, clearError } = useSimulation();

	return (
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
					<p>Interactive fire behavior modeling system</p>
				</div>
			</div>

			<div className="header-actions">
				<Link to="/" className="nav-link">
					<button className="btn btn-secondary" title="Home">
						<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
							<path d="M8.707 1.5a1 1 0 0 0-1.414 0L.646 8.146a.5.5 0 0 0 .708.708L8 2.207l6.646 6.647a.5.5 0 0 0 .708-.708L13 5.793V2.5a.5.5 0 0 0-.5-.5h-1a.5.5 0 0 0-.5.5v1.293L8.707 1.5Z" />
							<path d="m8 3.293 6 6V13.5a1.5 1.5 0 0 1-1.5 1.5h-9A1.5 1.5 0 0 1 2 13.5V9.293l6-6Z" />
						</svg>
						<span>Home</span>
					</button>
				</Link>

				<Link to="/help">
					<button className="btn btn-secondary" title="Documentation">
						<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
							<path d="M8 15A7 7 0 1 1 8 1a7 7 0 0 1 0 14zm0 1A8 8 0 1 0 8 0a8 8 0 0 0 0 16z" />
							<path d="M5.255 5.786a.237.237 0 0 0 .241.247h.825c.138 0 .248-.113.266-.25.09-.656.54-1.134 1.342-1.134.686 0 1.314.343 1.314 1.168 0 .635-.374.927-.965 1.371-.673.489-1.206 1.06-1.168 1.987l.003.217a.25.25 0 0 0 .25.246h.811a.25.25 0 0 0 .25-.25v-.105c0-.718.273-.927 1.01-1.486.609-.463 1.244-.977 1.244-2.056 0-1.511-1.276-2.241-2.673-2.241-1.267 0-2.655.59-2.75 2.286zm1.557 5.763c0 .533.425.927 1.01.927.609 0 1.028-.394 1.028-.927 0-.552-.42-.94-1.029-.94-.584 0-1.009.388-1.009.94z" />
						</svg>
						<span>Help</span>
					</button>
				</Link>

				<button
					className="btn btn-primary"
					disabled={!state.currentResults}
					onClick={() => {
						if (state.currentResults?.scenario_id) {
							window.open(`/api/export-results/${state.currentResults.scenario_id}`, "_blank");
						}
					}}
				>
					<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
						<path d="M15.964.686a.5.5 0 0 0-.65-.65L.767 5.855H.766l-.452.18a.5.5 0 0 0-.082.887l.41.26.001.002 4.995 3.178 3.178 4.995.002.002.26.41a.5.5 0 0 0 .886-.083l6-15Zm-1.833 1.89L6.637 10.07l-.215-.338a.5.5 0 0 0-.154-.154l-.338-.215 7.494-7.494 1.178-.471-.47 1.178Z" />
					</svg>
					<span>Export</span>
				</button>
			</div>

			{state.error && (
				<div className="alert alert-error">
					<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="currentColor" viewBox="0 0 16 16">
						<path d="M8 15A7 7 0 1 1 8 1a7 7 0 0 1 0 14zm0 1A8 8 0 1 0 8 0a8 8 0 0 0 0 16z" />
						<path d="M7.002 11a1 1 0 1 1 2 0 1 1 0 0 1-2 0zM7.1 4.995a.905.905 0 1 1 1.8 0l-.35 3.507a.552.552 0 0 1-1.1 0L7.1 4.995z" />
					</svg>
					<span>{state.error}</span>
					<button onClick={clearError} className="btn btn-icon">
						<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
							<path d="M4.646 4.646a.5.5 0 0 1 .708 0L8 7.293l2.646-2.647a.5.5 0 0 1 .708.708L8.707 8l2.647 2.646a.5.5 0 0 1-.708.708L8 8.707l-2.646 2.647a.5.5 0 0 1-.708-.708L7.293 8 4.646 5.354a.5.5 0 0 1 0-.708z" />
						</svg>
					</button>
				</div>
			)}

			{state.isLoading && (
				<div className="alert alert-info">
					<div className="loader-spinner"></div>
					<span>Running simulation, please wait...</span>
				</div>
			)}
		</header>
	);
};

export default Header;
