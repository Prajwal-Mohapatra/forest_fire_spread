// Chart Handler for Forest Fire Simulation
// Manages Chart.js visualizations for statistics and results

class FireSimCharts {
    constructor() {
        this.charts = {};
        this.chartData = {};
        this.chartOptions = {};
        this.initialized = false;
        
        this.init();
    }

    // Initialize chart system
    init() {
        // Wait for Chart.js to be available
        if (typeof Chart === 'undefined') {
            // Retry after a short delay
            setTimeout(() => this.init(), 100);
            return;
        }

        this.setupChartDefaults();
        this.initializeCharts();
        this.initialized = true;
        
        FireSimConfig.utils.log('info', 'Chart system initialized');
    }

    // Setup global Chart.js defaults
    setupChartDefaults() {
        Chart.defaults.font.family = 'Space Grotesk, system-ui, sans-serif';
        Chart.defaults.color = FireSimConfig.ui.colors.primary;
        Chart.defaults.backgroundColor = FireSimConfig.ui.colors.background;
        Chart.defaults.borderColor = FireSimConfig.ui.colors.border;
        
        // Set default responsive options
        Chart.defaults.responsive = true;
        Chart.defaults.maintainAspectRatio = false;
    }

    // Initialize all charts
    initializeCharts() {
        this.initializeBurnedAreaChart();
        this.initializeIntensityChart();
        this.initializeProgressChart();
        this.initializeComparisonChart();
    }

    // Initialize burned area over time chart
    initializeBurnedAreaChart() {
        const ctx = document.getElementById('area-chart');
        if (!ctx) return;

        const config = FireSimConfig.charts.areaChart;
        
        this.chartData.burnedArea = {
            labels: [],
            datasets: [{
                label: 'Burned Area (km²)',
                data: [],
                borderColor: config.color,
                backgroundColor: config.backgroundColor,
                fill: true,
                tension: 0.4,
                pointRadius: 4,
                pointHoverRadius: 6,
                pointBackgroundColor: config.color,
                pointBorderColor: '#ffffff',
                pointBorderWidth: 2
            }]
        };

        this.chartOptions.burnedArea = {
            ...FireSimConfig.charts.options,
            scales: {
                ...FireSimConfig.charts.options.scales,
                y: {
                    ...FireSimConfig.charts.options.scales.y,
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Area (km²)',
                        color: FireSimConfig.ui.colors.secondary
                    }
                },
                x: {
                    ...FireSimConfig.charts.options.scales.x,
                    title: {
                        display: true,
                        text: 'Time (hours)',
                        color: FireSimConfig.ui.colors.secondary
                    }
                }
            },
            plugins: {
                ...FireSimConfig.charts.options.plugins,
                title: {
                    display: true,
                    text: 'Fire Spread Area Over Time',
                    font: {
                        size: 16,
                        weight: 'bold'
                    },
                    color: FireSimConfig.ui.colors.primary
                }
            }
        };

        this.charts.burnedArea = new Chart(ctx, {
            type: 'line',
            data: this.chartData.burnedArea,
            options: this.chartOptions.burnedArea
        });
    }

    // Initialize fire intensity chart
    initializeIntensityChart() {
        const ctx = document.getElementById('intensity-chart');
        if (!ctx) return;

        const config = FireSimConfig.charts.intensityChart;
        
        this.chartData.intensity = {
            labels: [],
            datasets: [{
                label: 'Max Intensity (kW/m)',
                data: [],
                borderColor: config.color,
                backgroundColor: config.backgroundColor,
                fill: true,
                tension: 0.4,
                pointRadius: 4,
                pointHoverRadius: 6,
                pointBackgroundColor: config.color,
                pointBorderColor: '#ffffff',
                pointBorderWidth: 2
            }]
        };

        this.chartOptions.intensity = {
            ...FireSimConfig.charts.options,
            scales: {
                ...FireSimConfig.charts.options.scales,
                y: {
                    ...FireSimConfig.charts.options.scales.y,
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Intensity (kW/m)',
                        color: FireSimConfig.ui.colors.secondary
                    }
                },
                x: {
                    ...FireSimConfig.charts.options.scales.x,
                    title: {
                        display: true,
                        text: 'Time (hours)',
                        color: FireSimConfig.ui.colors.secondary
                    }
                }
            },
            plugins: {
                ...FireSimConfig.charts.options.plugins,
                title: {
                    display: true,
                    text: 'Fire Intensity Over Time',
                    font: {
                        size: 16,
                        weight: 'bold'
                    },
                    color: FireSimConfig.ui.colors.primary
                }
            }
        };

        this.charts.intensity = new Chart(ctx, {
            type: 'line',
            data: this.chartData.intensity,
            options: this.chartOptions.intensity
        });
    }

    // Initialize simulation progress chart
    initializeProgressChart() {
        const ctx = document.getElementById('progress-chart');
        if (!ctx) return;

        this.chartData.progress = {
            labels: ['Unburned', 'Burning', 'Burned'],
            datasets: [{
                data: [100, 0, 0],
                backgroundColor: [
                    '#00cc66',  // Green for unburned
                    '#ff6600',  // Orange for burning
                    '#666666'   // Gray for burned
                ],
                borderColor: '#ffffff',
                borderWidth: 2
            }]
        };

        this.chartOptions.progress = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 16,
                        usePointStyle: true,
                        font: {
                            size: 12
                        }
                    }
                },
                title: {
                    display: true,
                    text: 'Current Fire Status',
                    font: {
                        size: 16,
                        weight: 'bold'
                    },
                    color: FireSimConfig.ui.colors.primary
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const label = context.label || '';
                            const value = context.parsed || 0;
                            return `${label}: ${value.toFixed(1)}%`;
                        }
                    }
                }
            }
        };

        this.charts.progress = new Chart(ctx, {
            type: 'doughnut',
            data: this.chartData.progress,
            options: this.chartOptions.progress
        });
    }

    // Initialize scenario comparison chart
    initializeComparisonChart() {
        const ctx = document.getElementById('comparison-chart');
        if (!ctx) return;

        this.chartData.comparison = {
            labels: [],
            datasets: [
                {
                    label: 'Burned Area (km²)',
                    data: [],
                    backgroundColor: 'rgba(242, 115, 13, 0.7)',
                    borderColor: '#F2730D',
                    borderWidth: 2,
                    yAxisID: 'y'
                },
                {
                    label: 'Max Intensity (kW/m)',
                    data: [],
                    backgroundColor: 'rgba(220, 38, 127, 0.7)',
                    borderColor: '#dc267f',
                    borderWidth: 2,
                    yAxisID: 'y1'
                }
            ]
        };

        this.chartOptions.comparison = {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Scenarios',
                        color: FireSimConfig.ui.colors.secondary
                    }
                },
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: 'Burned Area (km²)',
                        color: FireSimConfig.ui.colors.secondary
                    }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Max Intensity (kW/m)',
                        color: FireSimConfig.ui.colors.secondary
                    },
                    grid: {
                        drawOnChartArea: false,
                    },
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Scenario Comparison',
                    font: {
                        size: 16,
                        weight: 'bold'
                    },
                    color: FireSimConfig.ui.colors.primary
                },
                legend: {
                    display: true,
                    position: 'top'
                }
            }
        };

        this.charts.comparison = new Chart(ctx, {
            type: 'bar',
            data: this.chartData.comparison,
            options: this.chartOptions.comparison
        });

        // Initially hide comparison chart
        this.hideChart('comparison');
    }

    // Update charts from simulation data
    async updateChartsFromSimulation(simulationData) {
        if (!this.initialized || !simulationData) return;

        try {
            const statistics = simulationData.hourly_statistics || [];
            
            // Update burned area chart
            this.updateBurnedAreaChart(statistics);
            
            // Update intensity chart
            this.updateIntensityChart(statistics);
            
            // Update progress chart with final frame data
            if (statistics.length > 0) {
                const latestStats = statistics[statistics.length - 1];
                this.updateProgressChart(latestStats);
            }

            FireSimConfig.utils.log('info', 'Charts updated from simulation data');

        } catch (error) {
            FireSimConfig.utils.log('error', 'Error updating charts:', error);
        }
    }

    // Update burned area chart
    updateBurnedAreaChart(statistics) {
        if (!this.charts.burnedArea) return;

        const labels = statistics.map((_, index) => index);
        const data = statistics.map(stat => stat.burned_area_km2 || 0);

        this.chartData.burnedArea.labels = labels;
        this.chartData.burnedArea.datasets[0].data = data;
        
        this.charts.burnedArea.update('none');
    }

    // Update intensity chart
    updateIntensityChart(statistics) {
        if (!this.charts.intensity) return;

        const labels = statistics.map((_, index) => index);
        const data = statistics.map(stat => stat.max_intensity || 0);

        this.chartData.intensity.labels = labels;
        this.chartData.intensity.datasets[0].data = data;
        
        this.charts.intensity.update('none');
    }

    // Update progress chart
    updateProgressChart(currentStats) {
        if (!this.charts.progress) return;

        const totalArea = currentStats.total_area || 1;
        const burnedArea = currentStats.burned_area_km2 || 0;
        const burningArea = currentStats.actively_burning_area || 0;
        const unburnedArea = Math.max(0, totalArea - burnedArea - burningArea);

        const totalSim = unburnedArea + burningArea + burnedArea;
        const unburnedPercent = (unburnedArea / totalSim) * 100;
        const burningPercent = (burningArea / totalSim) * 100;
        const burnedPercent = (burnedArea / totalSim) * 100;

        this.chartData.progress.datasets[0].data = [
            unburnedPercent,
            burningPercent,
            burnedPercent
        ];
        
        this.charts.progress.update('none');
    }

    // Update current frame data (for real-time updates during animation)
    updateCurrentFrameData(frameStats, frameIndex) {
        if (!this.initialized || !frameStats) return;

        // Update progress chart with current frame data
        this.updateProgressChart(frameStats);

        // Add current frame indicator to line charts
        this.addFrameIndicator('burnedArea', frameIndex);
        this.addFrameIndicator('intensity', frameIndex);
    }

    // Add frame indicator to line charts
    addFrameIndicator(chartName, frameIndex) {
        const chart = this.charts[chartName];
        if (!chart) return;

        // Remove existing indicator
        chart.data.datasets = chart.data.datasets.filter(dataset => dataset.label !== 'Current Frame');

        // Add new indicator
        const maxValue = Math.max(...chart.data.datasets[0].data);
        chart.data.datasets.push({
            label: 'Current Frame',
            data: [{x: frameIndex, y: maxValue * 1.1}],
            borderColor: '#0E88D3',
            backgroundColor: '#0E88D3',
            pointRadius: 8,
            pointStyle: 'triangle',
            showLine: false,
            order: 0
        });

        chart.update('none');
    }

    // Create scenario comparison
    createScenarioComparison(scenarios) {
        if (!this.charts.comparison) return;

        const labels = scenarios.map((_, index) => `Scenario ${index + 1}`);
        const areaData = scenarios.map(scenario => 
            scenario.final_statistics?.burned_area_km2 || 0
        );
        const intensityData = scenarios.map(scenario => 
            scenario.final_statistics?.max_intensity || 0
        );

        this.chartData.comparison.labels = labels;
        this.chartData.comparison.datasets[0].data = areaData;
        this.chartData.comparison.datasets[1].data = intensityData;

        this.charts.comparison.update();
        this.showChart('comparison');

        FireSimConfig.utils.log('info', 'Scenario comparison chart updated');
    }

    // Create custom chart for specific data
    createCustomChart(containerId, type, data, options = {}) {
        const ctx = document.getElementById(containerId);
        if (!ctx) return null;

        const customOptions = {
            ...FireSimConfig.charts.options,
            ...options
        };

        const chart = new Chart(ctx, {
            type: type,
            data: data,
            options: customOptions
        });

        return chart;
    }

    // Export chart as image
    exportChart(chartName, filename = null) {
        const chart = this.charts[chartName];
        if (!chart) return null;

        try {
            const base64Image = chart.toBase64Image();
            
            if (filename) {
                // Create download link
                const link = document.createElement('a');
                link.download = filename;
                link.href = base64Image;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            }

            return base64Image;

        } catch (error) {
            FireSimConfig.utils.log('error', 'Chart export failed:', error);
            return null;
        }
    }

    // Show/hide charts
    showChart(chartName) {
        const chart = this.charts[chartName];
        if (!chart) return;

        const container = chart.canvas.parentElement;
        if (container) {
            container.style.display = 'block';
        }
    }

    hideChart(chartName) {
        const chart = this.charts[chartName];
        if (!chart) return;

        const container = chart.canvas.parentElement;
        if (container) {
            container.style.display = 'none';
        }
    }

    // Clear all charts
    clearAllCharts() {
        Object.keys(this.charts).forEach(chartName => {
            this.clearChart(chartName);
        });
    }

    // Clear specific chart
    clearChart(chartName) {
        const chart = this.charts[chartName];
        if (!chart) return;

        // Reset data
        if (chart.data.labels) {
            chart.data.labels = [];
        }
        
        chart.data.datasets.forEach(dataset => {
            dataset.data = [];
        });

        chart.update();
    }

    // Resize charts (call when container size changes)
    resizeCharts() {
        Object.values(this.charts).forEach(chart => {
            if (chart && typeof chart.resize === 'function') {
                chart.resize();
            }
        });
    }

    // Update chart themes (for dark/light mode)
    updateTheme(isDark = false) {
        const textColor = isDark ? '#ffffff' : FireSimConfig.ui.colors.primary;
        const gridColor = isDark ? '#444444' : FireSimConfig.ui.colors.border;

        Object.values(this.charts).forEach(chart => {
            if (!chart) return;

            // Update options
            chart.options.plugins.title.color = textColor;
            chart.options.plugins.legend.labels.color = textColor;
            chart.options.scales.x.ticks.color = textColor;
            chart.options.scales.y.ticks.color = textColor;
            chart.options.scales.x.grid.color = gridColor;
            chart.options.scales.y.grid.color = gridColor;

            chart.update();
        });

        FireSimConfig.utils.log('info', `Charts theme updated to ${isDark ? 'dark' : 'light'} mode`);
    }

    // Add real-time data point
    addRealTimeDataPoint(chartName, label, value) {
        const chart = this.charts[chartName];
        if (!chart) return;

        chart.data.labels.push(label);
        chart.data.datasets[0].data.push(value);

        // Limit data points to prevent performance issues
        const maxPoints = 100;
        if (chart.data.labels.length > maxPoints) {
            chart.data.labels.shift();
            chart.data.datasets[0].data.shift();
        }

        chart.update('none');
    }

    // Get chart data for export
    getChartData(chartName) {
        const chart = this.charts[chartName];
        if (!chart) return null;

        return {
            labels: chart.data.labels,
            datasets: chart.data.datasets.map(dataset => ({
                label: dataset.label,
                data: dataset.data
            }))
        };
    }

    // Cleanup
    destroy() {
        Object.values(this.charts).forEach(chart => {
            if (chart && typeof chart.destroy === 'function') {
                chart.destroy();
            }
        });

        this.charts = {};
        this.chartData = {};
        this.chartOptions = {};
        this.initialized = false;

        FireSimConfig.utils.log('info', 'Chart system destroyed');
    }
}

// Create global charts instance
window.FireSimCharts = new FireSimCharts();

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = FireSimCharts;
}
