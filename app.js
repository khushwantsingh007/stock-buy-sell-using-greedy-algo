// Main app JavaScript file
const fetchBtn = document.getElementById('fetchBtn');
const tickerInput = document.getElementById('tickerInput');
const stockTitle = document.getElementById('stockTitle');
const lastUpdated = document.getElementById('lastUpdated');
const recommendationBadge = document.getElementById('recommendationBadge');
const recommendationText = document.getElementById('recommendationText');
const recommendationDate = document.getElementById('recommendationDate');
const signalExplanation = document.getElementById('signalExplanation');
const tradesTableBody = document.getElementById('tradesTableBody');

let currentTicker = '';
let autoRefreshInterval = null;
let chartInstances = {};

// Event listeners
fetchBtn.addEventListener('click', fetchStockData);
tickerInput.addEventListener('keypress', function(e) {
    if (e.key === 'Enter') fetchStockData();
});

// Main function to fetch stock data
function fetchStockData() {
    const ticker = tickerInput.value.trim().toUpperCase();
    
    if (!ticker || ticker.length > 15) {
        showAlert('Please enter a valid stock ticker (1-15 characters)', 'danger');
        return;
    }
    
    currentTicker = ticker;
    stockTitle.textContent = `${ticker} Analysis`;
    
    // Clear previous refresh interval if exists
    if (autoRefreshInterval) clearInterval(autoRefreshInterval);
    
    // Initial load
    loadData();
    
    // Set auto refresh
    autoRefreshInterval = setInterval(loadData, 60 * 1000); // Refresh every minute
}

// Function to load data from the API
function loadData() {
    showLoading(true);
    
    fetch(`/api/stock/${currentTicker}`)
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => {
                    throw new Error(err.error || `Error fetching data (Status: ${response.status})`);
                });
            }
            return response.json();
        })
        .then(data => {
            if (data.error) throw new Error(data.error);
            
            // Update UI with fetched data
            lastUpdated.textContent = `Last updated: ${data.last_updated}`;
            updateCharts(data);
            updateRecommendation(data.signals);
            updatePerformanceMetrics(data.signals.analysis);
            updateRecentTrades(data.signals.analysis.positions);
            
            // Show success message
            showAlert(`Successfully updated ${currentTicker} data`, 'success', 2000);
        })
        .catch(error => {
            console.error('Error:', error);
            showError(error.message);
        })
        .finally(() => showLoading(false));
}

// Function to update charts with new data
function updateCharts(data) {
    if (!data.prices || data.prices.length === 0) {
        console.error('No price data available');
        showAlert('No price data available for this ticker', 'warning');
        return;
    }

    try {
        const dates = data.prices.map(item => new Date(item.Date));
        const closes = data.prices.map(item => item.Close);

        // Get buy and sell signals
        const buySignals = data.signals.signals.filter(s => s.Signal === 'BUY');
        const sellSignals = data.signals.signals.filter(s => s.Signal === 'SELL');

        // Main price chart with signals
        Plotly.newPlot('mainChart', [
            {
                x: dates,
                y: closes,
                type: 'line',
                name: 'Price',
                line: {color: '#4cc9f0', width: 2}
            },
            {
                x: buySignals.map(s => new Date(s.Date)),
                y: buySignals.map(s => s.Price),
                mode: 'markers',
                name: 'Buy',
                marker: {
                    color: '#4ade80',
                    size: 10,
                    symbol: 'triangle-up'
                }
            },
            {
                x: sellSignals.map(s => new Date(s.Date)),
                y: sellSignals.map(s => s.Price),
                mode: 'markers',
                name: 'Sell',
                marker: {
                    color: '#f87171',
                    size: 10,
                    symbol: 'triangle-down'
                }
            }
        ], {
            title: `${currentTicker} Price with Buy/Sell Signals`,
            xaxis: {
                title: 'Date',
                color: '#e6e6e6',
                gridcolor: '#1a1a2e',
                zerolinecolor: '#1a1a2e'
            },
            yaxis: {
                title: 'Price',
                color: '#e6e6e6',
                gridcolor: '#1a1a2e',
                zerolinecolor: '#1a1a2e'
            },
            plot_bgcolor: '#16213e',
            paper_bgcolor: '#16213e',
            font: {color: '#e6e6e6'},
            showlegend: true,
            legend: {
                x: 0,
                y: 1,
                bgcolor: '#0f3460',
                bordercolor: '#1a1a2e'
            },
            margin: {t: 50, r: 50, l: 50, b: 50},
            autosize: true,
            responsive: true
        });

        // SMA chart
        if (data.indicators?.SMA_20 && data.indicators?.SMA_50) {
            Plotly.newPlot('smaChart', [
                {
                    x: dates,
                    y: closes,
                    type: 'line',
                    name: 'Price',
                    line: {color: '#4cc9f0', width: 1.5}
                },
                {
                    x: dates,
                    y: data.indicators.SMA_20,
                    type: 'line',
                    name: 'SMA 20',
                    line: {color: '#f59e0b', width: 2}
                },
                {
                    x: dates,
                    y: data.indicators.SMA_50,
                    type: 'line',
                    name: 'SMA 50',
                    line: {color: '#8b5cf6', width: 2}
                }
            ], {
                title: 'Moving Averages',
                xaxis: {
                    color: '#e6e6e6',
                    gridcolor: '#1a1a2e',
                    zerolinecolor: '#1a1a2e'
                },
                yaxis: {
                    color: '#e6e6e6',
                    gridcolor: '#1a1a2e',
                    zerolinecolor: '#1a1a2e'
                },
                plot_bgcolor: '#16213e',
                paper_bgcolor: '#16213e',
                font: {color: '#e6e6e6'},
                showlegend: true,
                legend: {
                    x: 0,
                    y: 1,
                    bgcolor: '#0f3460',
                    bordercolor: '#1a1a2e'
                },
                margin: {t: 50, r: 20, l: 50, b: 30},
                autosize: true
            });
        }

        // RSI chart
        if (data.indicators?.RSI) {
            Plotly.newPlot('rsiChart', [{
                x: dates,
                y: data.indicators.RSI,
                type: 'line',
                name: 'RSI',
                line: {color: '#10b981', width: 2}
            }], {
                title: 'RSI (14)',
                shapes: [
                    {
                        type: 'line', 
                        y0: 30, y1: 30, 
                        x0: dates[0], x1: dates[dates.length-1], 
                        line: {color: '#4ade80', dash: 'dash', width: 1}
                    },
                    {
                        type: 'line', 
                        y0: 70, y1: 70, 
                        x0: dates[0], x1: dates[dates.length-1], 
                        line: {color: '#f87171', dash: 'dash', width: 1}
                    }
                ],
                plot_bgcolor: '#16213e',
                paper_bgcolor: '#16213e',
                font: {color: '#e6e6e6'},
                xaxis: {
                    color: '#e6e6e6',
                    gridcolor: '#1a1a2e',
                    zerolinecolor: '#1a1a2e'
                },
                yaxis: {
                    range: [0, 100],
                    color: '#e6e6e6',
                    gridcolor: '#1a1a2e',
                    zerolinecolor: '#1a1a2e'
                },
                margin: {t: 50, r: 20, l: 50, b: 30},
                autosize: true
            });
        }

        // Make charts responsive
        window.addEventListener('resize', function() {
            Plotly.Plots.resize('mainChart');
            Plotly.Plots.resize('smaChart');
            Plotly.Plots.resize('rsiChart');
        });
    } catch (error) {
        console.error('Error updating charts:', error);
        showAlert('Error rendering charts. Please try again.', 'danger');
    }
}

// Function to update recommendation based on signals
function updateRecommendation(signals) {
    const rec = signals.recommendation || 'HOLD';
    recommendationText.textContent = rec;
    recommendationBadge.textContent = rec;
    recommendationDate.textContent = signals.recommendation_date || 'N/A';
    
    // Set recommendation styling and explanation
    if (rec === 'BUY') {
        recommendationBadge.className = 'badge bg-success';
        signalExplanation.innerHTML = `
            <strong>BUY Recommendation:</strong> 
            <p>The trading algorithm detected favorable conditions based on:</p>
            <ul>
                <li>RSI below 30 (oversold condition) indicating potential reversal</li>
                <li>20-day SMA crossing above 50-day SMA (bullish trend signal)</li>
            </ul>
            <p class="mb-0">Consider accumulating positions at current levels.</p>
        `;
        signalExplanation.className = 'alert alert-success';
    } else if (rec === 'SELL') {
        recommendationBadge.className = 'badge bg-danger';
        signalExplanation.innerHTML = `
            <strong>SELL Recommendation:</strong> 
            <p>The trading algorithm detected unfavorable conditions based on:</p>
            <ul>
                <li>RSI above 70 (overbought condition) indicating potential pullback</li>
                <li>20-day SMA crossing below 50-day SMA (bearish trend signal)</li>
            </ul>
            <p class="mb-0">Consider taking profits or reducing exposure.</p>
        `;
        signalExplanation.className = 'alert alert-danger';
    } else {
        recommendationBadge.className = 'badge bg-secondary';
        signalExplanation.innerHTML = `
            <strong>HOLD Recommendation:</strong> 
            <p>The trading algorithm didn't detect strong enough signals for action.</p>
            <p>Current market conditions appear neutral. Maintain current positions.</p>
            <p class="mb-0">RSI is between oversold and overbought levels, and moving averages are not showing clear crossover signals.</p>
        `;
        signalExplanation.className = 'alert alert-secondary';
    }
}

// Function to update performance metrics
function updatePerformanceMetrics(analysis) {
    document.getElementById('totalTrades').textContent = analysis.total_trades || 0;
    document.getElementById('profitableTrades').textContent = analysis.profitable_trades || 0;
    document.getElementById('winRate').textContent = analysis.win_rate ? `${analysis.win_rate}%` : '0%';
    document.getElementById('totalProfit').textContent = analysis.total_profit ? `$${analysis.total_profit.toFixed(2)}` : '$0.00';
}

// Function to update recent trades table
function updateRecentTrades(positions) {
    tradesTableBody.innerHTML = '';
    
    if (!positions || positions.length === 0) {
        tradesTableBody.innerHTML = `
            <tr>
                <td colspan="4" class="text-center">No trades yet</td>
            </tr>
        `;
        return;
    }
    
    // Sort positions by date (newest first)
    const sortedPositions = [...positions].sort((a, b) => {
        return new Date(b.date) - new Date(a.date);
    });
    
    sortedPositions.forEach(trade => {
        const row = document.createElement('tr');
        const date = trade.date ? new Date(trade.date).toLocaleDateString() : 'N/A';
        const profit = trade.profit !== undefined ? trade.profit.toFixed(2) : '-';
        const profitClass = trade.profit > 0 ? 'text-success' : (trade.profit < 0 ? 'text-danger' : '');
        
        row.innerHTML = `
            <td>${date}</td>
            <td><span class="badge ${trade.type === 'BUY' ? 'bg-success' : 'bg-danger'}">${trade.type}</span></td>
            <td>$${trade.price?.toFixed(2) || '0.00'}</td>
            <td class="${profitClass}">${profit !== '-' ? '$' + profit : profit}</td>
        `;
        tradesTableBody.appendChild(row);
    });
}

// Helper function to show loading state
function showLoading(show) {
    fetchBtn.disabled = show;
    fetchBtn.innerHTML = show 
        ? '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Loading...' 
        : 'Analyze';
}

// Helper function to show error
function showError(message) {
    showAlert(message, 'danger', 5000);
}

// Helper function to show alerts
function showAlert(message, type = 'info', duration = 5000) {
    // Remove existing alerts
    const existingAlerts = document.querySelectorAll('.alert-floating');
    existingAlerts.forEach(alert => {
        if (alert.dataset.type === type) {
            alert.remove();
        }
    });
    
    // Create new alert
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show alert-floating`;
    alertDiv.dataset.type = type;
    alertDiv.innerHTML = `
        ${type === 'danger' ? '<strong>Error:</strong> ' : ''}
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    // Add to document
    document.body.appendChild(alertDiv);
    
    // Auto-remove after duration
    if (duration > 0) {
        setTimeout(() => {
            alertDiv.classList.remove('show');
            setTimeout(() => alertDiv.remove(), 300);
        }, duration);
    }
}

// Initialize on page load
window.addEventListener('DOMContentLoaded', () => {
    // Add floating alert styling
    const style = document.createElement('style');
    style.textContent = `
        .alert-floating {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 1050;
            min-width: 250px;
            max-width: 400px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            border-left: 4px solid;
            animation: slideIn 0.3s ease-out forwards;
        }
        .alert-danger {
            border-left-color: #dc3545;
        }
        .alert-success {
            border-left-color: #198754;
        }
        .alert-warning {
            border-left-color: #ffc107;
        }
        .alert-info {
            border-left-color: #0dcaf0;
        }
        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
    `;
    document.head.appendChild(style);
    
    // Initialize with default ticker
    tickerInput.value = tickerInput.value || 'NSE:RELIANCE';
    currentTicker = tickerInput.value.trim().toUpperCase();
    fetchStockData();
});