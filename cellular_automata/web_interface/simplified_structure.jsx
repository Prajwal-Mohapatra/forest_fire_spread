// Simplified Forest Fire Simulation Interface Structure
// Main layout: Header + Two-column layout (Controls Sidebar + Main Content)

const FireSimulationInterface = () => (
  <div className="app-container">
    {/* Header */}
    <header className="header">
      <div className="team-info">
        <img src="logo.png" alt="Team Logo" />
        <h1>The Minions</h1>
      </div>
      <div className="hackathon-title">
        <h2>Bharatiya Antariksh Hackathon 2025</h2>
      </div>
    </header>

    {/* Main Content Area */}
    <div className="main-content">
      {/* Left Sidebar - Controls */}
      <aside className="controls-sidebar">
        
        {/* Ignition Controls */}
        <section className="ignition-controls">
          <h3>Ignition Controls</h3>
          <div className="toggle-switch">Click to Ignite Mode</div>
          
          <h4>Active Ignition Points</h4>
          <div className="ignition-point">Point 1 <button>×</button></div>
          <div className="ignition-point">Point 2 <button>×</button></div>
          <button className="clear-all-btn">Clear All Ignitions</button>
          
          <div className="slider-control">
            <label>Ignition Intensity: 50</label>
            <input type="range" />
          </div>
        </section>

        {/* Simulation Parameters */}
        <section className="sim-parameters">
          <h3>Simulation Parameters</h3>
          <div className="input-row">
            <input placeholder="Wind Speed (km/h)" />
            <input placeholder="Wind Direction (degrees)" />
          </div>
          
          <div className="slider-control">
            <label>Simulation Speed (x): 1</label>
            <input type="range" />
          </div>
          
          <select>Resolution</select>
          <select>Physics Model</select>
        </section>

        {/* Layer Toggles */}
        <section className="layer-toggles">
          <h3>Layer Toggles</h3>
          <label><input type="checkbox" /> Fire Probability Layer</label>
          <label><input type="checkbox" /> Current Fire Spread</label>
          <label><input type="checkbox" /> Terrain/DEM Overlay</label>
          <label><input type="checkbox" /> Roads/Barriers</label>
          <label><input type="checkbox" /> Weather Data Overlay</label>
        </section>

        {/* Animation Controls */}
        <section className="animation-controls">
          <h3>Animation Controls</h3>
          <div className="button-group">
            <button>Play</button>
            <button>Pause</button>
            <button>Stop</button>
          </div>
          
          <div className="slider-control">
            <label>Timeline Scrubber: 0</label>
            <input type="range" />
          </div>
          
          <div className="speed-buttons">
            <button>1x</button>
            <button>2x</button>
            <button>4x</button>
          </div>
          
          <div className="frame-controls">
            <button>Frame Back</button>
            <button>Frame Forward</button>
          </div>
        </section>
      </aside>

      {/* Main Visualization Area */}
      <main className="visualization-area">
        {/* Map/Simulation Display */}
        <div className="map-container">
          <img src="simulation-map.png" alt="Fire Simulation Map" />
        </div>

        {/* Progress Indicator */}
        <div className="progress-section">
          <h4>Simulation Progress</h4>
          <div className="progress-bar">
            <div className="progress-fill" style={{width: '60%'}}></div>
          </div>
          <span>60%</span>
        </div>

        {/* Timeline */}
        <div className="timeline">
          <div className="timeline-item">Start: 00:00</div>
          <div className="timeline-item">Ignition: 00:00</div>
          <div className="timeline-item">Rapid Spread: 00:00</div>
          <div className="timeline-item">Containment: 00:00</div>
          <div className="timeline-item">End: 00:00</div>
        </div>

        {/* Results Cards */}
        <section className="results-section">
          <h3>Simulation Results</h3>
          <div className="results-grid">
            <div className="result-card">
              <label>Total Area Burned</label>
              <value>1500 acres</value>
            </div>
            <div className="result-card">
              <label>Average Fire Intensity</label>
              <value>250 kW/m</value>
            </div>
            <div className="result-card">
              <label>Simulation Time</label>
              <value>12 hours</value>
            </div>
          </div>
        </section>

        {/* Statistics Charts */}
        <section className="statistics">
          <h3>Real-time Statistics</h3>
          <div className="charts-container">
            <img src="chart1.png" alt="Statistics Chart 1" />
            <img src="chart2.png" alt="Statistics Chart 2" />
          </div>
          <button className="export-btn">Export Results</button>
        </section>
      </main>
    </div>
  </div>
);

/* 
PROMPT-FRIENDLY SUMMARY:
Create a forest fire simulation web interface with:

1. Header: Team name "The Minions" + "Bharatiya Antariksh Hackathon 2025"

2. Two-column layout:
   - Left sidebar (320px) with controls
   - Main content area for visualization

3. Controls sidebar sections:
   - Ignition Controls: toggle switch, ignition points list, intensity slider
   - Simulation Parameters: wind inputs, speed slider, dropdowns for resolution/physics
   - Layer Toggles: checkboxes for map layers
   - Animation Controls: play/pause/stop buttons, timeline scrubber, speed buttons

4. Main content area:
   - Large map/simulation display
   - Progress bar (60% complete)
   - Timeline with key events
   - Results cards showing metrics (area burned, intensity, time)
   - Statistics charts
   - Export button

5. Color scheme: 
   - Background: #FCFAF7 (cream)
   - Buttons: #F5EDE8 (light peach)
   - Accent: #F2730D (orange)
   - Text: #1C140D (dark brown)
   - Font: Space Grotesk

6. Responsive design with flexbox layout
*/
