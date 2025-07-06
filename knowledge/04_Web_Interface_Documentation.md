# 🖥️ Web Interface Documentation

## Overview

The web interface provides an interactive demonstration platform for the forest fire spread simulation system, targeting ISRO researchers and forest fire management professionals. Built with a React frontend and Node.js backend, it offers real-time simulation control and visualization.

## Design Philosophy

### Target Audience
- **Primary**: ISRO researchers (10-15 years experience)
- **Secondary**: Forest fire management professionals
- **Context**: Scientific demonstration and hackathon evaluation

### Visual Theme
- **ISRO Aesthetic**: Deep space blues with ISRO orange accents
- **Scientific Interface**: Clean, data-driven, professional appearance
- **Fire Emergency Theme**: Strategic use of fire colors (red, orange, yellow)
- **Minimalistic Design**: Focus on functionality over decoration

## Architecture Overview

### Technology Stack

```
Frontend (React)
├── React 18+ with functional components
├── Material-UI for component library
├── Leaflet for interactive mapping
├── Chart.js for data visualization
└── WebSocket for real-time updates

Backend (Node.js)
├── Express.js REST API server
├── Socket.io for real-time communication
├── Python bridge for ML/CA integration
├── File system management for results
└── Authentication (optional for demo)

Integration Layer
├── ML-CA Bridge API calls
├── GeoTIFF processing and serving
├── Animation generation
└── Export functionality
```

### System Architecture

```
Web Browser ←→ React Frontend ←→ Node.js API ←→ Python ML/CA Bridge
     ↓              ↓                ↓              ↓
User Interface   Real-time       REST/WebSocket   ML Model
   Controls      Updates         Communication    CA Engine
```

## Frontend Components

### 1. Main Application Layout

```jsx
// App.jsx - Main application container
function App() {
  return (
    <div className="app-container">
      <Header />
      <div className="main-content">
        <ControlPanel />
        <MapDisplay />
      </div>
      <ResultsPanel />
      <StatusBar />
    </div>
  );
}
```

#### Layout Structure
```
┌─────────────────────────────────────────────────────────────┐
│ Header: ISRO Logo + Title + Navigation                     │
├───────────────────┬─────────────────────────────────────────┤
│ Control Panel     │ Interactive Map Display                 │
│ (25% width)       │ (75% width)                            │
│                   │                                         │
│ - Date Selection  │ - Leaflet map with fire layers         │
│ - Ignition Points │ - Click-to-ignite functionality        │
│ - Weather Params  │ - Animation controls                    │
│ - Simulation Ctrl │ - Zoom/pan controls                     │
│ - Layer Toggles   │ - Coordinate display                    │
│                   │                                         │
├───────────────────┴─────────────────────────────────────────┤
│ Results Panel (Collapsible)                                │
│ - Statistics, Timeline, Export Options                     │
├─────────────────────────────────────────────────────────────┤
│ Status Bar: Progress, Messages, Connection Status          │
└─────────────────────────────────────────────────────────────┘
```

### 2. Control Panel Components

#### Date and Time Selection
```jsx
// DateTimeControl.jsx
function DateTimeControl({ selectedDate, onDateChange, simulationHours, onHoursChange }) {
  return (
    <Card className="control-card">
      <CardHeader title="Date & Time Settings" />
      <CardContent>
        <DatePicker
          value={selectedDate}
          onChange={onDateChange}
          minDate="2016-04-01"
          maxDate="2016-05-29"
          format="YYYY-MM-DD"
        />
        
        <Select
          value={simulationHours}
          onChange={onHoursChange}
          label="Simulation Duration"
        >
          <MenuItem value={1}>1 Hour</MenuItem>
          <MenuItem value={2}>2 Hours</MenuItem>
          <MenuItem value={3}>3 Hours</MenuItem>
          <MenuItem value={6}>6 Hours</MenuItem>
          <MenuItem value={12}>12 Hours</MenuItem>
        </Select>
      </CardContent>
    </Card>
  );
}
```

#### Ignition Point Controls
```jsx
// IgnitionControl.jsx
function IgnitionControl({ ignitionPoints, onAddPoint, onRemovePoint, onClearAll }) {
  return (
    <Card className="control-card">
      <CardHeader title="Fire Ignition Points" />
      <CardContent>
        <FormControlLabel
          control={<Switch checked={ignitionMode} onChange={onIgnitionModeToggle} />}
          label="Click to Ignite Mode"
        />
        
        <List className="ignition-list">
          {ignitionPoints.map((point, index) => (
            <ListItem key={index}>
              <ListItemText 
                primary={`Point ${index + 1}`}
                secondary={`${point.lat.toFixed(4)}, ${point.lon.toFixed(4)}`}
              />
              <IconButton onClick={() => onRemovePoint(index)}>
                <DeleteIcon />
              </IconButton>
            </ListItem>
          ))}
        </List>
        
        <Button onClick={onClearAll} color="secondary" variant="outlined">
          Clear All Points
        </Button>
      </CardContent>
    </Card>
  );
}
```

#### Weather Parameters
```jsx
// WeatherControl.jsx
function WeatherControl({ weather, onWeatherChange }) {
  return (
    <Card className="control-card">
      <CardHeader title="Weather Conditions" />
      <CardContent>
        <Slider
          value={weather.windSpeed}
          onChange={(e, value) => onWeatherChange('windSpeed', value)}
          min={0}
          max={50}
          step={2.5}
          marks
          valueLabelDisplay="auto"
          aria-labelledby="wind-speed-slider"
        />
        
        <Select
          value={weather.windDirection}
          onChange={(e) => onWeatherChange('windDirection', e.target.value)}
          label="Wind Direction"
        >
          <MenuItem value={0}>North (0°)</MenuItem>
          <MenuItem value={45}>Northeast (45°)</MenuItem>
          <MenuItem value={90}>East (90°)</MenuItem>
          <MenuItem value={135}>Southeast (135°)</MenuItem>
          <MenuItem value={180}>South (180°)</MenuItem>
          <MenuItem value={225}>Southwest (225°)</MenuItem>
          <MenuItem value={270}>West (270°)</MenuItem>
          <MenuItem value={315}>Northwest (315°)</MenuItem>
        </Select>
        
        <TextField
          value={weather.temperature}
          onChange={(e) => onWeatherChange('temperature', e.target.value)}
          label="Temperature (°C)"
          type="number"
          inputProps={{ min: 15, max: 45 }}
        />
        
        <TextField
          value={weather.humidity}
          onChange={(e) => onWeatherChange('humidity', e.target.value)}
          label="Relative Humidity (%)"
          type="number"
          inputProps={{ min: 10, max: 80 }}
        />
      </CardContent>
    </Card>
  );
}
```

### 3. Interactive Map Display

#### Main Map Component
```jsx
// MapDisplay.jsx
import { MapContainer, TileLayer, GeoJSON, useMapEvents } from 'react-leaflet';

function MapDisplay({ ignitionPoints, onAddIgnition, fireLayer, probabilityLayer }) {
  const uttarakhandBounds = [
    [28.6, 77.8], // Southwest corner
    [31.1, 81.1]  // Northeast corner
  ];

  return (
    <div className="map-container">
      <MapContainer
        bounds={uttarakhandBounds}
        style={{ height: '100%', width: '100%' }}
        zoomControl={true}
      >
        {/* Base map layers */}
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution="&copy; OpenStreetMap contributors"
        />
        
        {/* Satellite imagery layer (optional) */}
        <TileLayer
          url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
          attribution="&copy; Esri"
        />
        
        {/* Fire probability heatmap layer */}
        {probabilityLayer && (
          <GeoJSON
            data={probabilityLayer}
            style={probabilityLayerStyle}
          />
        )}
        
        {/* Current fire spread layer */}
        {fireLayer && (
          <GeoJSON
            data={fireLayer}
            style={fireLayerStyle}
          />
        )}
        
        {/* Click handler for ignition points */}
        <ClickHandler onMapClick={onAddIgnition} />
        
        {/* Ignition point markers */}
        {ignitionPoints.map((point, index) => (
          <Marker
            key={index}
            position={[point.lat, point.lon]}
            icon={ignitionIcon}
          >
            <Popup>
              Ignition Point {index + 1}<br />
              {point.lat.toFixed(4)}, {point.lon.toFixed(4)}
            </Popup>
          </Marker>
        ))}
      </MapContainer>
      
      {/* Map controls overlay */}
      <div className="map-controls">
        <LayerControl />
        <AnimationControl />
        <CoordinateDisplay />
      </div>
    </div>
  );
}

// Click event handler component
function ClickHandler({ onMapClick }) {
  useMapEvents({
    click: (e) => {
      onMapClick(e.latlng);
    }
  });
  return null;
}
```

#### Layer Styling
```jsx
// Layer styling functions
const probabilityLayerStyle = (feature) => {
  const probability = feature.properties.probability;
  
  return {
    fillColor: getProbabilityColor(probability),
    weight: 0,
    opacity: 1,
    color: 'transparent',
    fillOpacity: 0.7
  };
};

const fireLayerStyle = (feature) => {
  const intensity = feature.properties.intensity;
  
  return {
    fillColor: getFireColor(intensity),
    weight: 1,
    opacity: 1,
    color: '#ff4444',
    fillOpacity: 0.8
  };
};

// Color mapping functions
function getProbabilityColor(probability) {
  if (probability < 0.2) return '#0066cc';      // Blue (low)
  if (probability < 0.4) return '#00cc66';      // Green (low-medium)
  if (probability < 0.6) return '#ffcc00';      // Yellow (medium)
  if (probability < 0.8) return '#ff6600';      // Orange (high)
  return '#cc0000';                             // Red (very high)
}

function getFireColor(intensity) {
  if (intensity < 0.2) return '#ff9999';        // Light red
  if (intensity < 0.4) return '#ff6666';        // Medium red
  if (intensity < 0.6) return '#ff3333';        // Red
  if (intensity < 0.8) return '#ff0000';        // Bright red
  return '#cc0000';                             // Dark red
}
```

### 4. Animation and Simulation Controls

#### Animation Control Panel
```jsx
// AnimationControl.jsx
function AnimationControl({ 
  isPlaying, onPlay, onPause, onStop, onStep,
  currentFrame, totalFrames, playSpeed, onSpeedChange 
}) {
  return (
    <Card className="animation-control">
      <CardContent>
        <div className="playback-controls">
          <IconButton onClick={onPlay} disabled={isPlaying}>
            <PlayArrowIcon />
          </IconButton>
          <IconButton onClick={onPause} disabled={!isPlaying}>
            <PauseIcon />
          </IconButton>
          <IconButton onClick={onStop}>
            <StopIcon />
          </IconButton>
          <IconButton onClick={onStep}>
            <SkipNextIcon />
          </IconButton>
        </div>
        
        <div className="timeline-control">
          <Slider
            value={currentFrame}
            min={0}
            max={totalFrames - 1}
            step={1}
            marks
            valueLabelDisplay="auto"
            valueLabelFormat={(value) => `Hour ${value}`}
          />
        </div>
        
        <div className="speed-control">
          <Typography>Speed:</Typography>
          <ToggleButtonGroup value={playSpeed} onChange={onSpeedChange} exclusive>
            <ToggleButton value={0.5}>0.5x</ToggleButton>
            <ToggleButton value={1}>1x</ToggleButton>
            <ToggleButton value={2}>2x</ToggleButton>
            <ToggleButton value={4}>4x</ToggleButton>
          </ToggleButtonGroup>
        </div>
      </CardContent>
    </Card>
  );
}
```

### 5. Results and Statistics Panel

#### Results Display Component
```jsx
// ResultsPanel.jsx
function ResultsPanel({ simulationResults, isOpen, onToggle }) {
  if (!simulationResults) return null;
  
  return (
    <Collapse in={isOpen}>
      <Card className="results-panel">
        <CardHeader 
          title="Simulation Results"
          action={
            <IconButton onClick={onToggle}>
              {isOpen ? <ExpandLessIcon /> : <ExpandMoreIcon />}
            </IconButton>
          }
        />
        
        <CardContent>
          <Grid container spacing={3}>
            {/* Statistics cards */}
            <Grid item xs={12} md={4}>
              <StatisticsCard 
                title="Burned Area"
                value={`${simulationResults.finalStats.burnedArea.toFixed(1)} ha`}
                icon={<LocalFireDepartmentIcon />}
                color="error"
              />
            </Grid>
            
            <Grid item xs={12} md={4}>
              <StatisticsCard 
                title="Maximum Intensity"
                value={simulationResults.finalStats.maxIntensity.toFixed(3)}
                icon={<WhatshotIcon />}
                color="warning"
              />
            </Grid>
            
            <Grid item xs={12} md={4}>
              <StatisticsCard 
                title="Simulation Time"
                value={`${simulationResults.duration} hours`}
                icon={<TimerIcon />}
                color="info"
              />
            </Grid>
            
            {/* Timeline chart */}
            <Grid item xs={12} md={8}>
              <TimelineChart data={simulationResults.hourlyStats} />
            </Grid>
            
            {/* Export options */}
            <Grid item xs={12} md={4}>
              <ExportOptions simulationId={simulationResults.scenarioId} />
            </Grid>
          </Grid>
        </CardContent>
      </Card>
    </Collapse>
  );
}
```

#### Statistics Visualization
```jsx
// TimelineChart.jsx
import { Line } from 'react-chartjs-2';

function TimelineChart({ data }) {
  const chartData = {
    labels: data.map((_, index) => `Hour ${index}`),
    datasets: [
      {
        label: 'Burned Area (ha)',
        data: data.map(d => d.burnedArea),
        borderColor: '#ff4444',
        backgroundColor: 'rgba(255, 68, 68, 0.1)',
        yAxisID: 'y'
      },
      {
        label: 'Fire Intensity',
        data: data.map(d => d.maxIntensity),
        borderColor: '#ff8800',
        backgroundColor: 'rgba(255, 136, 0, 0.1)',
        yAxisID: 'y1'
      }
    ]
  };
  
  const options = {
    responsive: true,
    interaction: {
      mode: 'index',
      intersect: false,
    },
    plugins: {
      title: {
        display: true,
        text: 'Fire Progression Over Time'
      }
    },
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: 'Time (hours)'
        }
      },
      y: {
        type: 'linear',
        display: true,
        position: 'left',
        title: {
          display: true,
          text: 'Burned Area (hectares)'
        }
      },
      y1: {
        type: 'linear',
        display: true,
        position: 'right',
        title: {
          display: true,
          text: 'Fire Intensity'
        },
        grid: {
          drawOnChartArea: false,
        },
      }
    }
  };
  
  return <Line data={chartData} options={options} />;
}
```

## Backend API Implementation

### Express.js Server Setup

```javascript
// server.js
const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const cors = require('cors');
const path = require('path');

const app = express();
const server = http.createServer(app);
const io = socketIo(server, {
  cors: {
    origin: "http://localhost:3000",
    methods: ["GET", "POST"]
  }
});

// Middleware
app.use(cors());
app.use(express.json());
app.use('/static', express.static(path.join(__dirname, 'public')));

// API Routes
app.use('/api/simulation', require('./routes/simulation'));
app.use('/api/data', require('./routes/data'));
app.use('/api/export', require('./routes/export'));

// WebSocket connection handling
io.on('connection', (socket) => {
  console.log('Client connected:', socket.id);
  
  socket.on('start-simulation', async (params) => {
    try {
      // Start simulation and emit progress updates
      await startSimulationWithProgress(params, (progress) => {
        socket.emit('simulation-progress', progress);
      });
    } catch (error) {
      socket.emit('simulation-error', { error: error.message });
    }
  });
  
  socket.on('disconnect', () => {
    console.log('Client disconnected:', socket.id);
  });
});

const PORT = process.env.PORT || 5000;
server.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
```

### API Route Implementations

#### Simulation Routes
```javascript
// routes/simulation.js
const express = require('express');
const { spawn } = require('child_process');
const router = express.Router();

// Start new simulation
router.post('/start', async (req, res) => {
  const { dateStr, ignitionPoints, weatherParams, simulationHours } = req.body;
  
  try {
    // Validate parameters
    if (!dateStr || !ignitionPoints || ignitionPoints.length === 0) {
      return res.status(400).json({ error: 'Missing required parameters' });
    }
    
    // Call Python ML-CA bridge
    const pythonProcess = spawn('python', [
      'scripts/run_simulation.py',
      '--date', dateStr,
      '--ignition-points', JSON.stringify(ignitionPoints),
      '--weather', JSON.stringify(weatherParams),
      '--hours', simulationHours.toString()
    ]);
    
    let outputData = '';
    let errorData = '';
    
    pythonProcess.stdout.on('data', (data) => {
      outputData += data.toString();
    });
    
    pythonProcess.stderr.on('data', (data) => {
      errorData += data.toString();
    });
    
    pythonProcess.on('close', (code) => {
      if (code === 0) {
        try {
          const result = JSON.parse(outputData);
          res.json({
            status: 'success',
            scenarioId: result.scenario_id,
            results: result
          });
        } catch (parseError) {
          res.status(500).json({ error: 'Failed to parse simulation results' });
        }
      } else {
        res.status(500).json({ 
          error: 'Simulation failed',
          details: errorData
        });
      }
    });
    
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Get simulation status
router.get('/status/:scenarioId', (req, res) => {
  const { scenarioId } = req.params;
  
  // Check simulation status and return current state
  // Implementation depends on how simulation state is tracked
  
  res.json({
    scenarioId,
    status: 'completed', // or 'running', 'failed'
    progress: 100,
    currentHour: 6
  });
});

// Get simulation results
router.get('/results/:scenarioId', (req, res) => {
  const { scenarioId } = req.params;
  
  try {
    // Load results from file system
    const resultsPath = path.join(__dirname, '../simulation_outputs', scenarioId, 'results.json');
    const results = JSON.parse(fs.readFileSync(resultsPath, 'utf8'));
    
    res.json(results);
  } catch (error) {
    res.status(404).json({ error: 'Results not found' });
  }
});

module.exports = router;
```

#### Data Routes
```javascript
// routes/data.js
const express = require('express');
const fs = require('fs');
const path = require('path');
const router = express.Router();

// Get available dates
router.get('/dates', (req, res) => {
  const dates = [
    '2016-04-01', '2016-04-02', '2016-04-03', // ... full date range
    '2016-05-27', '2016-05-28', '2016-05-29'
  ];
  
  res.json({ dates });
});

// Serve simulation frames
router.get('/frames/:scenarioId/:frameNumber', (req, res) => {
  const { scenarioId, frameNumber } = req.params;
  
  const framePath = path.join(
    __dirname, '../simulation_outputs', 
    scenarioId, 'frames', `frame_${frameNumber}.tif`
  );
  
  if (fs.existsSync(framePath)) {
    res.sendFile(path.resolve(framePath));
  } else {
    res.status(404).json({ error: 'Frame not found' });
  }
});

// Serve probability maps
router.get('/probability/:date', (req, res) => {
  const { date } = req.params;
  
  const probPath = path.join(
    __dirname, '../ml_outputs', `probability_${date}.tif`
  );
  
  if (fs.existsSync(probPath)) {
    res.sendFile(path.resolve(probPath));
  } else {
    res.status(404).json({ error: 'Probability map not found' });
  }
});

module.exports = router;
```

## State Management

### React Context for Global State

```jsx
// context/SimulationContext.js
import React, { createContext, useContext, useReducer } from 'react';

const SimulationContext = createContext();

const initialState = {
  // Simulation parameters
  selectedDate: '2016-05-15',
  simulationHours: 6,
  ignitionPoints: [],
  weatherParams: {
    windSpeed: 15,
    windDirection: 225,
    temperature: 30,
    humidity: 40
  },
  
  // Simulation state
  isRunning: false,
  currentResults: null,
  currentFrame: 0,
  
  // UI state
  ignitionMode: false,
  layersVisible: {
    probability: true,
    fire: true,
    terrain: false
  }
};

function simulationReducer(state, action) {
  switch (action.type) {
    case 'SET_DATE':
      return { ...state, selectedDate: action.payload };
      
    case 'ADD_IGNITION_POINT':
      return {
        ...state,
        ignitionPoints: [...state.ignitionPoints, action.payload]
      };
      
    case 'REMOVE_IGNITION_POINT':
      return {
        ...state,
        ignitionPoints: state.ignitionPoints.filter((_, index) => index !== action.payload)
      };
      
    case 'CLEAR_IGNITION_POINTS':
      return { ...state, ignitionPoints: [] };
      
    case 'UPDATE_WEATHER':
      return {
        ...state,
        weatherParams: { ...state.weatherParams, [action.key]: action.value }
      };
      
    case 'START_SIMULATION':
      return { ...state, isRunning: true, currentResults: null };
      
    case 'SIMULATION_COMPLETE':
      return { ...state, isRunning: false, currentResults: action.payload };
      
    case 'SET_CURRENT_FRAME':
      return { ...state, currentFrame: action.payload };
      
    default:
      return state;
  }
}

export function SimulationProvider({ children }) {
  const [state, dispatch] = useReducer(simulationReducer, initialState);
  
  return (
    <SimulationContext.Provider value={{ state, dispatch }}>
      {children}
    </SimulationContext.Provider>
  );
}

export function useSimulation() {
  const context = useContext(SimulationContext);
  if (!context) {
    throw new Error('useSimulation must be used within SimulationProvider');
  }
  return context;
}
```

## Real-time Communication

### WebSocket Integration

```jsx
// hooks/useWebSocket.js
import { useEffect, useRef } from 'react';
import io from 'socket.io-client';

export function useWebSocket(url, onMessage) {
  const socketRef = useRef();
  
  useEffect(() => {
    socketRef.current = io(url);
    
    socketRef.current.on('simulation-progress', (progress) => {
      onMessage('progress', progress);
    });
    
    socketRef.current.on('simulation-complete', (results) => {
      onMessage('complete', results);
    });
    
    socketRef.current.on('simulation-error', (error) => {
      onMessage('error', error);
    });
    
    return () => {
      socketRef.current.disconnect();
    };
  }, [url, onMessage]);
  
  const sendMessage = (event, data) => {
    if (socketRef.current) {
      socketRef.current.emit(event, data);
    }
  };
  
  return { sendMessage };
}
```

## Styling and Theme

### Material-UI Theme Configuration

```jsx
// theme/theme.js
import { createTheme } from '@mui/material/styles';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1E3A5F',      // ISRO deep blue
      light: '#4A6FA5',
      dark: '#0B1426'
    },
    secondary: {
      main: '#FF6B35',      // ISRO orange
      light: '#FF8C42',
      dark: '#E55A2B'
    },
    error: {
      main: '#D63031',      // Fire red
      light: '#E17055'
    },
    warning: {
      main: '#FDCB6E',      // Fire yellow
      dark: '#F39C12'
    },
    success: {
      main: '#00B894',      // Forest green
      dark: '#6C5CE7'
    },
    background: {
      default: '#F8F9FA',
      paper: '#FFFFFF'
    }
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h4: {
      fontWeight: 600,
      color: '#1E3A5F'
    },
    h6: {
      fontWeight: 500,
      color: '#2D3436'
    }
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
          borderRadius: 8
        }
      }
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          borderRadius: 6
        }
      }
    }
  }
});

export default theme;
```

### Custom CSS Styles

```css
/* styles/App.css */
.app-container {
  height: 100vh;
  display: flex;
  flex-direction: column;
  font-family: 'Inter', sans-serif;
}

.main-content {
  flex: 1;
  display: flex;
  overflow: hidden;
}

.control-panel {
  width: 25%;
  min-width: 300px;
  background: #f8f9fa;
  border-right: 1px solid #dee2e6;
  overflow-y: auto;
  padding: 16px;
}

.map-container {
  flex: 1;
  position: relative;
}

.control-card {
  margin-bottom: 16px !important;
}

.animation-control {
  position: absolute;
  bottom: 20px;
  left: 20px;
  right: 20px;
  z-index: 1000;
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(5px);
}

.results-panel {
  background: white;
  border-top: 1px solid #dee2e6;
}

.status-bar {
  height: 40px;
  background: #1E3A5F;
  color: white;
  display: flex;
  align-items: center;
  padding: 0 16px;
  font-size: 14px;
}

/* Fire intensity color classes */
.fire-intensity-0 { background-color: rgba(255, 153, 153, 0.7); }
.fire-intensity-1 { background-color: rgba(255, 102, 102, 0.7); }
.fire-intensity-2 { background-color: rgba(255, 51, 51, 0.7); }
.fire-intensity-3 { background-color: rgba(255, 0, 0, 0.7); }
.fire-intensity-4 { background-color: rgba(204, 0, 0, 0.7); }

/* Probability color classes */
.probability-0 { background-color: rgba(0, 102, 204, 0.7); }
.probability-1 { background-color: rgba(0, 204, 102, 0.7); }
.probability-2 { background-color: rgba(255, 204, 0, 0.7); }
.probability-3 { background-color: rgba(255, 102, 0, 0.7); }
.probability-4 { background-color: rgba(204, 0, 0, 0.7); }
```

## Deployment Configuration

### Production Build Setup

```json
// package.json
{
  "name": "forest-fire-simulation-web",
  "version": "1.0.0",
  "private": true,
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "@mui/material": "^5.14.0",
    "@mui/icons-material": "^5.14.0",
    "leaflet": "^1.9.0",
    "react-leaflet": "^4.2.0",
    "chart.js": "^4.3.0",
    "react-chartjs-2": "^5.2.0",
    "socket.io-client": "^4.7.0",
    "axios": "^1.4.0"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject"
  },
  "proxy": "http://localhost:5000"
}
```

### Docker Configuration

```dockerfile
# Dockerfile
FROM node:18-alpine as build

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

---

**Key Components:**
- React frontend with Material-UI components
- Interactive Leaflet map with fire visualization
- Real-time WebSocket communication
- Node.js/Express backend API
- Professional ISRO-themed interface

**Integration Points:**
- ML-CA Bridge: Python subprocess calls for simulation execution
- File System: GeoTIFF serving and animation generation
- Real-time Updates: WebSocket progress notifications
- Export System: Results download and visualization
