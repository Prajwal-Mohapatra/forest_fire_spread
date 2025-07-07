// Configuration for Forest Fire Simulation Web Application
// The Minions - Bharatiya Antariksh Hackathon 2025

window.FireSimConfig = {
    // API Configuration
    api: {
        baseUrl: 'http://localhost:5000/api',
        timeout: 30000, // 30 seconds
        retryAttempts: 3,
        retryDelay: 1000, // 1 second
        endpoints: {
            health: '/health',
            simulate: '/simulate',
            status: '/simulation/{id}/status',
            animation: '/simulation/{id}/animation',
            frame: '/simulation/{id}/frame/{hour}',
            dates: '/available_dates',
            config: '/config',
            multipleScenarios: '/multiple-scenarios',
            cache: '/simulation-cache/{id}',
            export: '/export-results/{id}'
        }
    },

    // Map Configuration
    map: {
        // Uttarakhand bounds
        bounds: {
            north: 31.1,
            south: 28.6,
            east: 81.1,
            west: 77.8
        },
        center: {
            lat: 30.3165,
            lon: 78.0322
        },
        zoom: {
            initial: 8,
            min: 6,
            max: 16
        },
        // Map layers
        layers: {
            satellite: {
                url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attribution: '&copy; Esri'
            },
            street: {
                url: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
                attribution: '&copy; OpenStreetMap contributors'
            },
            terrain: {
                url: 'https://stamen-tiles-{s}.a.ssl.fastly.net/terrain/{z}/{x}/{y}.{ext}',
                attribution: 'Map tiles by Stamen Design, CC BY 3.0 &mdash; Map data &copy; OpenStreetMap contributors',
                subdomains: 'abcd',
                ext: 'png'
            }
        }
    },

    // Simulation Parameters
    simulation: {
        // Default weather parameters
        defaultWeather: {
            windDirection: 45,    // degrees
            windSpeed: 15,        // km/h
            temperature: 30,      // Celsius
            humidity: 40          // percentage
        },
        
        // Parameter ranges
        ranges: {
            windDirection: { min: 0, max: 360, step: 15 },
            windSpeed: { min: 0, max: 100, step: 2.5 },
            temperature: { min: 15, max: 45, step: 1 },
            humidity: { min: 10, max: 80, step: 5 },
            ignitionIntensity: { min: 10, max: 100, step: 5 },
            simulationSpeed: { min: 0.5, max: 4, step: 0.5 }
        },

        // Duration options (in hours)
        durationOptions: [1, 2, 3, 6, 12],
        
        // Default duration
        defaultDuration: 6,
        
        // Maximum ignition points
        maxIgnitionPoints: 10,
        
        // Animation settings
        animation: {
            defaultSpeed: 1000,  // milliseconds per frame
            speedMultipliers: {
                '0.5x': 2000,
                '1x': 1000,
                '2x': 500,
                '4x': 250
            }
        }
    },

    // UI Configuration
    ui: {
        // Theme colors (matching sample_style.md)
        colors: {
            primary: '#1C140D',
            secondary: '#9C704A',
            accent: '#F2730D',
            background: '#FCFAF7',
            backgroundSection: '#F5EDE8',
            border: '#E8D9CF',
            highlight: '#0E88D3'
        },
        
        // Animation timings
        animations: {
            fast: 150,
            normal: 300,
            slow: 500
        },
        
        // Breakpoints for responsive design
        breakpoints: {
            mobile: 480,
            tablet: 768,
            desktop: 1024,
            wide: 1200
        },
        
        // Notification settings
        notifications: {
            duration: 5000,
            position: 'top-right'
        }
    },

    // Layer Configuration
    layers: {
        probability: {
            id: 'fire-probability',
            name: 'Fire Probability Layer',
            enabled: true,
            opacity: 0.7,
            colors: {
                veryLow: '#0066cc',    // Blue
                low: '#00cc66',        // Green
                medium: '#ffcc00',     // Yellow
                high: '#ff6600',       // Orange
                veryHigh: '#cc0000'    // Red
            }
        },
        fireSpread: {
            id: 'fire-spread',
            name: 'Current Fire Spread',
            enabled: true,
            opacity: 0.8,
            colors: {
                burning: '#ff4444',
                burned: '#666666',
                extinguished: '#999999'
            }
        },
        terrain: {
            id: 'terrain',
            name: 'Terrain/DEM Overlay',
            enabled: false,
            opacity: 0.5
        },
        barriers: {
            id: 'barriers',
            name: 'Roads/Barriers',
            enabled: false,
            opacity: 0.8
        },
        weather: {
            id: 'weather',
            name: 'Weather Data Overlay',
            enabled: false,
            opacity: 0.6
        }
    },

    // Demo Scenarios
    demos: {
        dehradun: {
            name: 'Dehradun Valley Fire',
            icon: '🏔️',
            description: 'Simulates fire spread in Dehradun valley terrain',
            parameters: {
                ignitionPoints: [
                    { lat: 30.3165, lon: 78.0322 }
                ],
                weather: {
                    windDirection: 225,
                    windSpeed: 20,
                    temperature: 35,
                    humidity: 30
                },
                duration: 6,
                date: '2016-05-15'
            }
        },
        rishikesh: {
            name: 'Rishikesh Forest Fire',
            icon: '🌊',
            description: 'Forest fire near Rishikesh with river barriers',
            parameters: {
                ignitionPoints: [
                    { lat: 30.0869, lon: 78.2676 }
                ],
                weather: {
                    windDirection: 45,
                    windSpeed: 15,
                    temperature: 32,
                    humidity: 45
                },
                duration: 8,
                date: '2016-05-20'
            }
        },
        nainital: {
            name: 'Nainital Hill Fire',
            icon: '⛰️',
            description: 'High altitude fire simulation near Nainital',
            parameters: {
                ignitionPoints: [
                    { lat: 29.3803, lon: 79.4636 }
                ],
                weather: {
                    windDirection: 180,
                    windSpeed: 25,
                    temperature: 28,
                    humidity: 35
                },
                duration: 4,
                date: '2016-05-25'
            }
        }
    },

    // Chart Configuration
    charts: {
        areaChart: {
            type: 'line',
            title: 'Burned Area Over Time',
            yAxis: 'Area (km²)',
            color: '#F2730D',
            backgroundColor: 'rgba(242, 115, 13, 0.1)'
        },
        intensityChart: {
            type: 'line',
            title: 'Fire Intensity Over Time',
            yAxis: 'Intensity',
            color: '#dc267f',
            backgroundColor: 'rgba(220, 38, 127, 0.1)'
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'white',
                    titleColor: '#1C140D',
                    bodyColor: '#1C140D',
                    borderColor: '#E8D9CF',
                    borderWidth: 1
                }
            },
            scales: {
                x: {
                    grid: {
                        color: '#E8D9CF',
                        borderColor: '#E8D9CF'
                    },
                    ticks: {
                        color: '#9C704A'
                    }
                },
                y: {
                    grid: {
                        color: '#E8D9CF',
                        borderColor: '#E8D9CF'
                    },
                    ticks: {
                        color: '#9C704A'
                    }
                }
            }
        }
    },

    // Export Configuration
    export: {
        formats: [
            { id: 'json', name: 'JSON Data', extension: '.json' },
            { id: 'csv', name: 'CSV Spreadsheet', extension: '.csv' },
            { id: 'geotiff', name: 'GeoTIFF Images', extension: '.tif' },
            { id: 'animation', name: 'Animation GIF', extension: '.gif' }
        ],
        defaultFormat: 'json'
    },

    // Error Messages
    errors: {
        api: {
            connection: 'Unable to connect to simulation server',
            timeout: 'Request timed out - please try again',
            server: 'Server error occurred',
            notFound: 'Requested resource not found',
            validation: 'Invalid input parameters'
        },
        simulation: {
            noIgnition: 'Please add at least one ignition point',
            invalidCoordinates: 'Ignition point coordinates are invalid',
            invalidWeather: 'Weather parameters are out of range',
            failed: 'Simulation execution failed',
            interrupted: 'Simulation was interrupted'
        },
        map: {
            loadFailed: 'Failed to load map tiles',
            outOfBounds: 'Location is outside the simulation area'
        }
    },

    // Success Messages
    messages: {
        simulation: {
            started: 'Simulation started successfully',
            completed: 'Simulation completed successfully',
            exported: 'Results exported successfully'
        },
        demo: {
            loaded: 'Demo scenario loaded',
            started: 'Demo simulation started'
        }
    },

    // Development/Debug Settings
    debug: {
        enabled: false, // Set to true for development
        logLevel: 'info', // 'debug', 'info', 'warn', 'error'
        mockData: false, // Use mock data when API is unavailable
        skipAnimations: false // Skip animations for testing
    },

    // Feature Flags
    features: {
        realTimeUpdates: true,
        multipleScenarios: true,
        exportOptions: true,
        weatherOverlay: true,
        terrainLayer: true,
        animation: true,
        charts: true
    },

    // Performance Settings
    performance: {
        maxFrames: 24, // Maximum frames for animation
        chartUpdateInterval: 1000, // Chart update frequency (ms)
        mapUpdateThrottle: 250, // Map update throttling (ms)
        debounceDelay: 300 // Input debounce delay (ms)
    }
};

// Utility functions for configuration
window.FireSimConfig.utils = {
    // Get API endpoint URL with parameters
    getApiUrl: function(endpoint, params = {}) {
        let url = this.api.baseUrl + this.api.endpoints[endpoint];
        
        // Replace path parameters
        Object.keys(params).forEach(key => {
            url = url.replace(`{${key}}`, params[key]);
        });
        
        return url;
    },

    // Validate coordinates within Uttarakhand bounds
    validateCoordinates: function(lat, lon) {
        const bounds = this.map.bounds;
        return lat >= bounds.south && lat <= bounds.north && 
               lon >= bounds.west && lon <= bounds.east;
    },

    // Get color for probability value
    getProbabilityColor: function(probability) {
        if (probability < 0.2) return this.layers.probability.colors.veryLow;
        if (probability < 0.4) return this.layers.probability.colors.low;
        if (probability < 0.6) return this.layers.probability.colors.medium;
        if (probability < 0.8) return this.layers.probability.colors.high;
        return this.layers.probability.colors.veryHigh;
    },

    // Format time for display
    formatTime: function(minutes) {
        const hours = Math.floor(minutes / 60);
        const mins = minutes % 60;
        return `${hours.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}`;
    },

    // Debounce function for performance
    debounce: function(func, delay) {
        let timeoutId;
        return function (...args) {
            clearTimeout(timeoutId);
            timeoutId = setTimeout(() => func.apply(this, args), delay);
        };
    },

    // Throttle function for performance
    throttle: function(func, limit) {
        let inThrottle;
        return function (...args) {
            if (!inThrottle) {
                func.apply(this, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    },

    // Log function with level control
    log: function(level, message, data = null) {
        if (!this.debug.enabled) return;
        
        const levels = ['debug', 'info', 'warn', 'error'];
        const currentLevelIndex = levels.indexOf(this.debug.logLevel);
        const messageLevelIndex = levels.indexOf(level);
        
        if (messageLevelIndex >= currentLevelIndex) {
            const timestamp = new Date().toISOString();
            console[level](`[${timestamp}] [FireSim] ${message}`, data || '');
        }
    }
};

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = window.FireSimConfig;
}
