# Forest Fire Simulation Web Interface

A comprehensive web-based interface for forest fire simulation and visualization, developed by **The Minions** team for the Bharatiya Antariksh Hackathon 2025.

> **New: React.js Implementation!** 🚀 The web interface has been completely redesigned and rebuilt using React.js and Leaflet to provide an even more dynamic and responsive user experience. See the [React Frontend](#react-frontend) section below for details.

## Overview

This web interface provides an intuitive platform for simulating forest fire spread using machine learning models and cellular automata algorithms. It features real-time visualization, interactive mapping, and comprehensive analytics. The system is now available in two versions:

1. **Original Flask-based UI** (Legacy version)
2. **Modern React.js UI** (New version)

## Features

### 🗺️ Interactive Map

- **Leaflet-based mapping** with multiple base layers (Street, Satellite, Terrain)
- **Click-to-add ignition points** with visual markers
- **Real-time fire spread visualization** with animated overlays
- **Layer toggles** for probability, terrain, barriers, and weather data
- **Wind direction indicator** with dynamic updates

### 🎮 Simulation Controls

- **Weather parameter adjustment** (wind speed/direction, temperature, humidity)
- **Multiple ignition points** support (up to 10 points)
- **Simulation duration** control (1-12 hours)
- **Physics model selection** (ML-enhanced or traditional CA)
- **Real-time progress tracking** with detailed status updates

### 📊 Visualization & Analytics

- **Real-time charts** using Chart.js for burned area and fire intensity
- **Progress pie chart** showing current fire status distribution
- **Timeline visualization** with key milestones
- **Scenario comparison** charts for multiple simulation runs
- **Export functionality** for results in multiple formats (JSON, CSV, GeoTIFF)

### 🚀 Demo Scenarios

- **Dehradun Valley Fire**: Mountain terrain simulation
- **Rishikesh Forest Fire**: River barrier effects
- **Nainital Hill Fire**: High altitude fire behavior

### 🎬 Animation System

- **Playback controls** (Play, Pause, Stop, Frame stepping)
- **Variable speed** options (0.5x to 4x)
- **Timeline scrubbing** for precise frame navigation
- **Smooth transitions** between simulation frames

## File Structure

```
web_interface/
├── app.py                          # Flask application entry point
├── app_with_react.py               # Flask configured to serve React app
├── api.py                          # Backend API endpoints
├── deploy.sh                       # React deployment script
├── test_frontend.py                # Frontend testing script
├── templates/
│   └── index.html                  # Main HTML template (legacy)
├── static/
│   ├── css/
│   │   ├── main.css               # Main styles and layout (legacy)
│   │   └── components.css         # Component-specific styles (legacy)
│   ├── js/
│   │   ├── config.js              # Configuration and constants (legacy)
│   │   ├── api.js                 # API service layer (legacy)
│   │   ├── ui-components.js       # UI interaction logic (legacy)
│   │   ├── map-handler.js         # Leaflet map management (legacy)
│   │   ├── simulation-manager.js  # Simulation orchestration (legacy)
│   │   ├── chart-handler.js       # Chart.js visualization (legacy)
│   │   └── main.js                # Application initialization (legacy)
│   └── images/
│       └── logo.svg               # Team logo
├── frontend/                       # React.js application (new)
│   ├── public/                     # Static assets for React
│   ├── src/                        # React source code
│   │   ├── components/             # UI components
│   │   ├── context/                # State management
│   │   ├── pages/                  # Page components
│   │   ├── services/               # API services
│   │   └── styles/                 # CSS stylesheets
│   └── package.json                # Dependencies and scripts
```

## Technology Stack

### Frontend (Legacy UI)

- **HTML5/CSS3** with modern grid and flexbox layouts
- **Vanilla JavaScript** (ES6+) for maximum performance
- **Leaflet.js** for interactive mapping
- **Chart.js** for data visualization
- **Space Grotesk** font for modern typography

### Frontend (React UI)

- **React.js** for component-based UI development
- **React Router** for client-side routing
- **React Leaflet** for interactive mapping
- **Recharts** for data visualization
- **Axios** for API communication
- **CSS Modules** for component-scoped styling

### Backend

- **Flask** web framework
- **Python 3.8+** for API endpoints
- **RESTful API** design with JSON responses

### Dependencies

#### Legacy UI

- **Leaflet 1.9.4** - Interactive maps
- **Chart.js 4.4.0** - Data visualization
- **Font CDN** - Space Grotesk typography

#### React UI

- **React 18.2.0** - UI library
- **React Leaflet 4.2.1** - Map integration
- **Leaflet 1.9.4** - Mapping library
- **Axios 1.4.0** - HTTP client
- **React Router 6.14.1** - Routing
- **Recharts 2.7.2** - Data visualization

## Quick Start

### Legacy UI

#### 1. Start the Application (Legacy UI)

```bash
cd cellular_automata/web_interface
python app.py
```

#### 2. Open in Browser

Navigate to `http://localhost:5000/old` for the legacy UI

#### 3. Run a Demo

- Click on any demo scenario button
- Watch as ignition points and parameters are automatically set
- Click "Start Simulation" to begin

#### 4. Manual Simulation

- Enable "Ignite Mode" toggle
- Click on the map to add ignition points
- Adjust weather parameters as needed
- Click "Start Simulation"

### React UI

#### 1. Build and Deploy React UI

```bash
cd cellular_automata/web_interface
./deploy.sh
python app.py
```

#### 2. Open in Browser

Navigate to `http://localhost:5000` for the new React UI

#### 3. Using the React UI

- Select a date from the dropdown
- Adjust weather parameters using the sliders
- Click on the map to add ignition points
- Click "Run Simulation" to start
- View the animated results and statistics

## Configuration

All configuration is centralized in `static/js/config.js`:

```javascript
window.FireSimConfig = {
	api: {
		baseUrl: "http://localhost:5000/api",
		timeout: 30000,
		retryAttempts: 3,
	},
	map: {
		center: { lat: 30.3165, lon: 78.0322 }, // Uttarakhand
		zoom: { initial: 8, min: 6, max: 16 },
	},
	simulation: {
		maxIgnitionPoints: 10,
		defaultDuration: 6,
	},
	// ... more configuration options
};
```

## API Endpoints

- `GET /` - Main application page
- `GET /api/health` - Health check
- `GET /api/available_dates` - Get available simulation dates
- `GET /api/config` - Get API configuration
- `POST /api/simulate` - Start new simulation
- `GET /api/simulation/{id}/status` - Get simulation status
- `GET /api/simulation/{id}/animation` - Get animation data
- `GET /api/export-results/{id}` - Export simulation results
- `GET /api/simulation/{id}/export-csv` - Download CSV report of simulation results
- `GET /api/simulation/{id}/download-animation` - Download animation as GIF or ZIP (use `?format=zip` for ZIP)

## Browser Compatibility

- **Chrome** 90+ ✅
- **Firefox** 88+ ✅
- **Safari** 14+ ✅
- **Edge** 90+ ✅

## Performance Features

- **Modular architecture** for optimal loading
- **Debounced inputs** to prevent excessive API calls
- **Chart animation optimization** with reduced motion support
- **Map layer caching** for improved performance
- **Responsive design** for mobile and desktop

## Error Handling

- **Comprehensive error messages** for user guidance
- **Automatic retry logic** for network requests
- **Graceful degradation** when API is unavailable
- **Browser compatibility warnings** for unsupported features

## Development

### Testing

Run the test suite:

```bash
python test_frontend.py
```

### Debugging

Enable debug mode in `config.js`:

```javascript
debug: {
    enabled: true,
    logLevel: 'debug'
}
```

### CSS Architecture

- **CSS Custom Properties** for consistent theming
- **Mobile-first** responsive design
- **Component-based** styling approach
- **Accessibility** considerations with ARIA labels

### JavaScript Architecture

- **Class-based modules** for maintainability
- **Event-driven** architecture
- **Promise-based** async operations
- **Error boundary** patterns

## React Frontend

The new React-based frontend provides a modern, component-based implementation of the Forest Fire Simulation interface with improved interactivity and responsiveness.

### Key Features of the React Implementation

- **Component-Based Architecture**: Modular design with reusable components
- **Context API State Management**: Centralized simulation state
- **React Leaflet Integration**: Enhanced mapping capabilities
- **Responsive Design**: Optimized for all device sizes
- **Modern UI**: ISRO/fire-themed interface with consistent styling
- **Client-Side Routing**: Seamless page transitions
- **Documentation Page**: Comprehensive help and instructions

### Quick Start (React UI)

#### Development Mode

1. Install dependencies:

```bash
cd cellular_automata/web_interface/frontend
npm install
```

2. Start development server:

```bash
npm start
```

3. Access development UI at `http://localhost:3000`

#### Production Deployment

1. Build and deploy the React frontend:

```bash
cd cellular_automata/web_interface
chmod +x deploy.sh
./deploy.sh
```

2. Start the Flask server:

```bash
python app.py
```

3. Access the app at `http://localhost:5000`

### React Component Structure

```
frontend/src/
├── components/          # UI components
│   ├── Header.js        # Application header with navigation
│   ├── ControlPanel.js  # Simulation controls and parameters
│   ├── MapInterface.js  # Leaflet map integration
│   └── ResultsPanel.js  # Simulation results and charts
├── context/
│   └── SimulationContext.js  # Global state management
├── pages/
│   └── Documentation.js      # Help and documentation
├── services/
│   └── api.js                # API communication layer
└── styles/                  # CSS stylesheets
```

## Contributing

1. Follow the existing code style and patterns
2. Test changes with the provided test script
3. Update documentation for new features
4. Ensure cross-browser compatibility
5. For React components, follow the component structure

## License

Developed for the Bharatiya Antariksh Hackathon 2025.

---

**The Minions Team** 🔥
_Advancing forest fire simulation technology_
