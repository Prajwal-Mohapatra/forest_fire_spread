# Forest Fire Simulation Web Interface

A comprehensive web-based interface for forest fire simulation and visualization, developed by **The Minions** team for the Bharatiya Antariksh Hackathon 2025.

## Overview

This web interface provides an intuitive platform for simulating forest fire spread using machine learning models and cellular automata algorithms. It features real-time visualization, interactive mapping, and comprehensive analytics.

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
├── api.py                          # Backend API endpoints
├── test_frontend.py                # Frontend testing script
├── templates/
│   └── index.html                  # Main HTML template
├── static/
│   ├── css/
│   │   ├── main.css               # Main styles and layout
│   │   └── components.css         # Component-specific styles
│   ├── js/
│   │   ├── config.js              # Configuration and constants
│   │   ├── api.js                 # API service layer
│   │   ├── ui-components.js       # UI interaction logic
│   │   ├── map-handler.js         # Leaflet map management
│   │   ├── simulation-manager.js  # Simulation orchestration
│   │   ├── chart-handler.js       # Chart.js visualization
│   │   └── main.js                # Application initialization
│   └── images/
│       └── logo.svg               # Team logo
```

## Technology Stack

### Frontend
- **HTML5/CSS3** with modern grid and flexbox layouts
- **Vanilla JavaScript** (ES6+) for maximum performance
- **Leaflet.js** for interactive mapping
- **Chart.js** for data visualization
- **Space Grotesk** font for modern typography

### Backend
- **Flask** web framework
- **Python 3.8+** for API endpoints
- **RESTful API** design with JSON responses

### Dependencies
- **Leaflet 1.9.4** - Interactive maps
- **Chart.js 4.4.0** - Data visualization
- **Font CDN** - Space Grotesk typography

## Quick Start

### 1. Start the Application
```bash
cd cellular_automata/web_interface
python app.py
```

### 2. Open in Browser
Navigate to `http://localhost:5000`

### 3. Run a Demo
- Click on any demo scenario button
- Watch as ignition points and parameters are automatically set
- Click "Start Simulation" to begin

### 4. Manual Simulation
- Enable "Ignite Mode" toggle
- Click on the map to add ignition points
- Adjust weather parameters as needed
- Click "Start Simulation"

## Configuration

All configuration is centralized in `static/js/config.js`:

```javascript
window.FireSimConfig = {
    api: {
        baseUrl: 'http://localhost:5000/api',
        timeout: 30000,
        retryAttempts: 3
    },
    map: {
        center: { lat: 30.3165, lon: 78.0322 }, // Uttarakhand
        zoom: { initial: 8, min: 6, max: 16 }
    },
    simulation: {
        maxIgnitionPoints: 10,
        defaultDuration: 6
    }
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

## Contributing

1. Follow the existing code style and patterns
2. Test changes with the provided test script
3. Update documentation for new features
4. Ensure cross-browser compatibility

## License

Developed for the Bharatiya Antariksh Hackathon 2025.

---

**The Minions Team** 🔥
*Advancing forest fire simulation technology*
