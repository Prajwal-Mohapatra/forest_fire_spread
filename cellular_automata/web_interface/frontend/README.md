# Forest Fire Simulation React Frontend

This directory contains the React.js implementation of the Forest Fire Spread Simulation web interface. The application provides an interactive interface for simulating forest fire spread using real-world data and machine learning models.

## Technology Stack

- **React.js**: Frontend library for building UI components
- **Leaflet**: Interactive mapping and visualization
- **React Router**: Client-side routing
- **Recharts**: Data visualization and charting
- **Axios**: API communication
- **Framer Motion**: Animation effects

## Features

- Interactive map interface with Leaflet
- Real-time fire spread simulation
- Weather parameter controls
- Animation playback of simulation results
- Statistical analysis and visualization
- Multiple base map layers (Satellite, Street, Terrain)
- Responsive design for various device sizes

## Directory Structure

```
frontend/
├── public/              # Static assets and HTML template
├── src/
│   ├── components/      # React components
│   ├── context/         # State management
│   ├── services/        # API services
│   ├── styles/          # CSS stylesheets
│   ├── App.js           # Main app component
│   └── index.js         # Entry point
├── package.json         # Dependencies and scripts
└── .env                 # Environment variables
```

## Setup and Installation

### Development

1. Install dependencies:

   ```
   npm install
   ```

2. Start the development server:

   ```
   npm start
   ```

3. Access the application at `http://localhost:3000`

### Production Build

1. Create an optimized build:

   ```
   npm run build
   ```

2. The build files will be in the `build/` directory

## Integration with Flask Backend

This React application is designed to work with the Flask API backend. The communication happens through API endpoints defined in the `src/services/api.js` file.

To deploy the React frontend with the Flask backend:

1. Build the React application
2. Run the deployment script from the parent directory:

   ```
   ./deploy.sh
   ```

3. This will configure the Flask app to serve the React build

## Environment Variables

- `REACT_APP_API_BASE_URL`: Base URL for API requests
- `REACT_APP_WEBSOCKET_URL`: URL for WebSocket connections (future use)

## Notes for Developers

- All API calls are abstracted in the `services/api.js` file
- Global state is managed through React Context in `context/SimulationContext.js`
- Map interactions are handled in the `MapInterface` component
- The styling follows the ISRO/fire theme as specified in the design documents
