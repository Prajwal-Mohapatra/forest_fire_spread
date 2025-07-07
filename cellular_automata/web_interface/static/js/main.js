// Main Application Entry Point for Forest Fire Simulation
// Initializes all modules and wires up the complete application

class FireSimApplication {
    constructor() {
        this.modules = {};
        this.initialized = false;
        this.loadingSteps = [
            'Initializing configuration...',
            'Setting up API service...',
            'Initializing user interface...',
            'Loading map components...',
            'Setting up chart system...',
            'Initializing simulation manager...',
            'Connecting demo scenarios...',
            'Performing health checks...',
            'Application ready!'
        ];
        this.currentStep = 0;
        
        this.init();
    }

    // Initialize the complete application
    async init() {
        try {
            this.showLoadingScreen();
            await this.initializeModules();
            await this.setupDemoHandlers();
            await this.performHealthChecks();
            this.hideLoadingScreen();
            this.showWelcomeMessage();
            
            this.initialized = true;
            FireSimConfig.utils.log('info', 'Fire Simulation Application initialized successfully');
            
        } catch (error) {
            this.handleInitializationError(error);
        }
    }

    // Show application loading screen
    showLoadingScreen() {
        const loadingHtml = `
            <div id="app-loading" style="
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: linear-gradient(135deg, #FCFAF7 0%, #F5EDE8 100%);
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                z-index: 10000;
                font-family: 'Space Grotesk', system-ui, sans-serif;
            ">
                <div style="text-align: center; max-width: 400px;">
                    <div style="font-size: 64px; margin-bottom: 24px;">🔥</div>
                    <h1 style="
                        color: #1C140D;
                        font-size: 28px;
                        font-weight: 700;
                        margin: 0 0 12px 0;
                    ">Forest Fire Simulation</h1>
                    <p style="
                        color: #9C704A;
                        font-size: 16px;
                        margin: 0 0 32px 0;
                    ">The Minions - Bharatiya Antariksh Hackathon 2025</p>
                    
                    <div style="
                        width: 100%;
                        height: 6px;
                        background: #E8D9CF;
                        border-radius: 3px;
                        overflow: hidden;
                        margin-bottom: 16px;
                    ">
                        <div id="app-loading-progress" style="
                            width: 0%;
                            height: 100%;
                            background: linear-gradient(90deg, #F2730D 0%, #9C704A 100%);
                            transition: width 0.3s ease;
                        "></div>
                    </div>
                    
                    <div id="app-loading-text" style="
                        color: #1C140D;
                        font-size: 14px;
                        font-weight: 500;
                    ">Initializing...</div>
                </div>
            </div>
        `;
        
        document.body.insertAdjacentHTML('afterbegin', loadingHtml);
    }

    // Update loading progress
    updateLoadingProgress(step, message) {
        this.currentStep = step;
        const progress = (step / this.loadingSteps.length) * 100;
        
        const progressBar = document.getElementById('app-loading-progress');
        const loadingText = document.getElementById('app-loading-text');
        
        if (progressBar) {
            progressBar.style.width = `${progress}%`;
        }
        
        if (loadingText) {
            loadingText.textContent = message;
        }
    }

    // Hide loading screen
    hideLoadingScreen() {
        const loadingElement = document.getElementById('app-loading');
        if (loadingElement) {
            loadingElement.style.transition = 'opacity 0.5s ease';
            loadingElement.style.opacity = '0';
            setTimeout(() => {
                loadingElement.remove();
            }, 500);
        }
    }

    // Initialize all application modules
    async initializeModules() {
        // Step 1: Configuration (already loaded)
        this.updateLoadingProgress(1, this.loadingSteps[0]);
        await this.delay(200);

        // Step 2: API Service
        this.updateLoadingProgress(2, this.loadingSteps[1]);
        this.modules.api = window.FireSimAPI;
        if (!this.modules.api) {
            throw new Error('API service not available');
        }
        await this.delay(300);

        // Step 3: User Interface
        this.updateLoadingProgress(3, this.loadingSteps[2]);
        this.modules.ui = window.FireSimUI;
        if (!this.modules.ui) {
            throw new Error('UI components not available');
        }
        await this.delay(400);

        // Step 4: Map Handler
        this.updateLoadingProgress(4, this.loadingSteps[3]);
        this.modules.map = window.FireSimMap;
        if (!this.modules.map) {
            throw new Error('Map handler not available');
        }
        await this.delay(500);

        // Step 5: Chart System
        this.updateLoadingProgress(5, this.loadingSteps[4]);
        this.modules.charts = window.FireSimCharts;
        if (!this.modules.charts) {
            throw new Error('Chart system not available');
        }
        await this.delay(300);

        // Step 6: Simulation Manager
        this.updateLoadingProgress(6, this.loadingSteps[5]);
        this.modules.manager = window.FireSimManager;
        if (!this.modules.manager) {
            throw new Error('Simulation manager not available');
        }
        await this.delay(400);

        FireSimConfig.utils.log('info', 'All modules initialized');
    }

    // Setup demo scenario handlers
    async setupDemoHandlers() {
        this.updateLoadingProgress(7, this.loadingSteps[6]);
        
        // Setup demo buttons
        Object.keys(FireSimConfig.demos).forEach(demoKey => {
            const demo = FireSimConfig.demos[demoKey];
            const button = document.querySelector(`[data-demo="${demoKey}"]`);
            
            if (button) {
                button.addEventListener('click', () => {
                    this.runDemo(demoKey);
                });
                
                // Update button text with demo info
                button.innerHTML = `
                    <span class="demo-icon">${demo.icon}</span>
                    <div class="demo-info">
                        <div class="demo-name">${demo.name}</div>
                        <div class="demo-desc">${demo.description}</div>
                    </div>
                `;
            }
        });

        // If no demo buttons found, create them
        if (document.querySelectorAll('[data-demo]').length === 0) {
            this.createDemoButtons();
        }

        await this.delay(300);
    }

    // Create demo buttons if they don't exist
    createDemoButtons() {
        const demoContainer = document.querySelector('.demo-scenarios') || 
                            document.querySelector('#demo-container');
        
        if (!demoContainer) return;

        Object.keys(FireSimConfig.demos).forEach(demoKey => {
            const demo = FireSimConfig.demos[demoKey];
            const button = document.createElement('button');
            button.className = 'btn-demo';
            button.dataset.demo = demoKey;
            button.innerHTML = `
                <span class="demo-icon">${demo.icon}</span>
                <div class="demo-info">
                    <div class="demo-name">${demo.name}</div>
                    <div class="demo-desc">${demo.description}</div>
                </div>
            `;
            
            button.addEventListener('click', () => {
                this.runDemo(demoKey);
            });
            
            demoContainer.appendChild(button);
        });
    }

    // Run demo scenario
    runDemo(demoKey) {
        if (!this.initialized) {
            FireSimConfig.utils.log('warn', 'Application not fully initialized');
            return;
        }

        if (this.modules.manager) {
            this.modules.manager.runDemoScenario(demoKey);
        }
    }

    // Perform health checks
    async performHealthChecks() {
        this.updateLoadingProgress(8, this.loadingSteps[7]);
        
        try {
            // Check API health
            const healthResult = await this.modules.api.checkHealth();
            if (!healthResult.success) {
                FireSimConfig.utils.log('warn', 'API health check failed:', healthResult.error);
                this.showApiWarning();
            } else {
                FireSimConfig.utils.log('info', 'API health check passed');
            }

            // Check required DOM elements
            this.checkRequiredElements();

            // Check browser compatibility
            this.checkBrowserCompatibility();

        } catch (error) {
            FireSimConfig.utils.log('warn', 'Health checks completed with warnings:', error);
        }

        await this.delay(500);
        this.updateLoadingProgress(9, this.loadingSteps[8]);
        await this.delay(300);
    }

    // Check required DOM elements
    checkRequiredElements() {
        const requiredElements = [
            'map-display',
            'ignition-points-list',
            'start-simulation-btn',
            'area-chart',
            'intensity-chart'
        ];

        const missing = requiredElements.filter(id => !document.getElementById(id));
        
        if (missing.length > 0) {
            FireSimConfig.utils.log('warn', 'Missing DOM elements:', missing);
        }
    }

    // Check browser compatibility
    checkBrowserCompatibility() {
        const checks = {
            fetch: typeof fetch !== 'undefined',
            localStorage: typeof localStorage !== 'undefined',
            canvas: !!document.createElement('canvas').getContext,
            webgl: !!document.createElement('canvas').getContext('webgl'),
            flexbox: CSS.supports('display', 'flex')
        };

        const failed = Object.keys(checks).filter(check => !checks[check]);
        
        if (failed.length > 0) {
            FireSimConfig.utils.log('warn', 'Browser compatibility issues:', failed);
            this.showCompatibilityWarning(failed);
        }
    }

    // Show API warning
    showApiWarning() {
        const warningHtml = `
            <div id="api-warning" style="
                position: fixed;
                top: 20px;
                right: 20px;
                background: #fff3cd;
                border: 1px solid #ffeaa7;
                color: #856404;
                padding: 12px 16px;
                border-radius: 6px;
                font-size: 14px;
                max-width: 300px;
                z-index: 9999;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            ">
                ⚠️ <strong>API Warning:</strong> Backend server may not be available. Some features may not work properly.
                <button onclick="this.parentElement.remove()" style="
                    float: right;
                    background: none;
                    border: none;
                    color: #856404;
                    cursor: pointer;
                    margin-left: 8px;
                ">×</button>
            </div>
        `;
        
        document.body.insertAdjacentHTML('beforeend', warningHtml);
        
        // Auto-remove after 10 seconds
        setTimeout(() => {
            const warning = document.getElementById('api-warning');
            if (warning) warning.remove();
        }, 10000);
    }

    // Show compatibility warning
    showCompatibilityWarning(issues) {
        const warningHtml = `
            <div id="compat-warning" style="
                position: fixed;
                top: 20px;
                left: 20px;
                background: #f8d7da;
                border: 1px solid #f1aeb5;
                color: #721c24;
                padding: 12px 16px;
                border-radius: 6px;
                font-size: 14px;
                max-width: 300px;
                z-index: 9999;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            ">
                🚫 <strong>Browser Compatibility:</strong> Your browser may not support all features. Issues: ${issues.join(', ')}
                <button onclick="this.parentElement.remove()" style="
                    float: right;
                    background: none;
                    border: none;
                    color: #721c24;
                    cursor: pointer;
                    margin-left: 8px;
                ">×</button>
            </div>
        `;
        
        document.body.insertAdjacentHTML('beforeend', warningHtml);
    }

    // Show welcome message
    showWelcomeMessage() {
        if (this.modules.ui) {
            setTimeout(() => {
                const message = 'Welcome to Forest Fire Simulation! Click on the map to add ignition points and start simulating.';
                this.modules.ui.showSuccess(message);
            }, 1000);
        }
    }

    // Handle initialization errors
    handleInitializationError(error) {
        this.hideLoadingScreen();
        
        const errorHtml = `
            <div id="init-error" style="
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: #FCFAF7;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                z-index: 10000;
                font-family: 'Space Grotesk', system-ui, sans-serif;
                text-align: center;
                padding: 40px;
            ">
                <div style="font-size: 64px; margin-bottom: 24px;">⚠️</div>
                <h1 style="color: #dc3545; font-size: 24px; margin-bottom: 16px;">
                    Application Failed to Initialize
                </h1>
                <p style="color: #6c757d; font-size: 16px; margin-bottom: 24px; max-width: 500px;">
                    ${error.message || 'An unknown error occurred during initialization.'}
                </p>
                <button onclick="window.location.reload()" style="
                    background: #F2730D;
                    color: white;
                    border: none;
                    padding: 12px 24px;
                    border-radius: 6px;
                    font-size: 16px;
                    cursor: pointer;
                    margin-right: 12px;
                ">Reload Application</button>
                <button onclick="this.parentElement.remove()" style="
                    background: transparent;
                    color: #6c757d;
                    border: 1px solid #6c757d;
                    padding: 12px 24px;
                    border-radius: 6px;
                    font-size: 16px;
                    cursor: pointer;
                ">Continue Anyway</button>
            </div>
        `;
        
        document.body.insertAdjacentHTML('afterbegin', errorHtml);
        
        FireSimConfig.utils.log('error', 'Application initialization failed:', error);
    }

    // Utility: delay function
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    // Setup keyboard shortcuts
    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + Enter: Start simulation
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                e.preventDefault();
                if (this.modules.manager) {
                    this.modules.manager.startSimulation();
                }
            }

            // Space: Play/Pause animation
            if (e.key === ' ' && !e.target.matches('input, textarea, select')) {
                e.preventDefault();
                if (this.modules.ui) {
                    if (this.modules.ui.state.animationState === 'playing') {
                        this.modules.ui.pauseAnimation();
                    } else {
                        this.modules.ui.playAnimation();
                    }
                }
            }

            // Escape: Clear ignition points
            if (e.key === 'Escape') {
                if (this.modules.ui) {
                    this.modules.ui.clearIgnitionPoints();
                }
            }

            // R: Reset simulation
            if (e.key === 'r' || e.key === 'R') {
                if (e.ctrlKey || e.metaKey) {
                    e.preventDefault();
                    this.resetApplication();
                }
            }
        });

        FireSimConfig.utils.log('info', 'Keyboard shortcuts activated');
    }

    // Reset application state
    resetApplication() {
        if (this.modules.ui) {
            this.modules.ui.clearIgnitionPoints();
            this.modules.ui.stopAnimation();
        }
        
        if (this.modules.charts) {
            this.modules.charts.clearAllCharts();
        }
        
        if (this.modules.manager) {
            this.modules.manager.stopAnimation();
            this.modules.manager.currentSimulation = null;
        }

        FireSimConfig.utils.log('info', 'Application state reset');
    }

    // Get application status
    getStatus() {
        return {
            initialized: this.initialized,
            modules: Object.keys(this.modules).reduce((status, key) => {
                status[key] = !!this.modules[key];
                return status;
            }, {}),
            simulationState: this.modules.manager ? this.modules.manager.getSimulationState() : null
        };
    }

    // Export application data
    async exportApplicationData() {
        const status = this.getStatus();
        const simulationData = this.modules.manager ? this.modules.manager.simulationData : null;
        const chartData = {};
        
        if (this.modules.charts) {
            Object.keys(this.modules.charts.charts).forEach(chartName => {
                chartData[chartName] = this.modules.charts.getChartData(chartName);
            });
        }

        const exportData = {
            timestamp: new Date().toISOString(),
            version: '1.0.0',
            status,
            simulationData,
            chartData,
            configuration: FireSimConfig
        };

        return exportData;
    }

    // Cleanup and destroy application
    destroy() {
        // Cleanup modules in reverse order
        if (this.modules.manager) {
            this.modules.manager.destroy();
        }
        
        if (this.modules.charts) {
            this.modules.charts.destroy();
        }
        
        if (this.modules.map) {
            this.modules.map.destroy();
        }
        
        if (this.modules.ui) {
            this.modules.ui.destroy();
        }

        this.modules = {};
        this.initialized = false;
        
        FireSimConfig.utils.log('info', 'Application destroyed');
    }
}

// Initialize application when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.FireSimApp = new FireSimApplication();
    
    // Setup keyboard shortcuts after initialization
    setTimeout(() => {
        if (window.FireSimApp.initialized) {
            window.FireSimApp.setupKeyboardShortcuts();
        }
    }, 2000);
});

// Handle page unload
window.addEventListener('beforeunload', () => {
    if (window.FireSimApp) {
        window.FireSimApp.destroy();
    }
});

// Handle visibility change
document.addEventListener('visibilitychange', () => {
    if (document.hidden && window.FireSimApp && window.FireSimApp.modules.ui) {
        // Pause animation when tab is hidden
        if (window.FireSimApp.modules.ui.state.animationState === 'playing') {
            window.FireSimApp.modules.ui.pauseAnimation();
        }
    }
});

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = FireSimApplication;
}
