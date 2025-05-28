# 📈 Stock Buy/Sell Recommendation System 
*A greedy algorithm-based trading advisor*

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.0-green)](https://flask.palletsprojects.com/)

## 🔍 Overview
Real-time stock analysis system that:
- Fetches NSE/BSE data using Yahoo Finance API
- Generates signals using **greedy algorithm** with RSI/SMA/MACD
- Visualizes trends with Plotly.js
- Provides actionable BUY/SELL recommendations

## 🚀 Features
- Multi-indicator scoring system
- Risk management (stop-loss/take-profit)
- Interactive backtesting charts
- Responsive web interface

## ⚙️ Installation
```bash
# Clone repo
git clone https://github.com/yourusername/Stock-Buy-Sell-Greedy-Algorithm.git

# Setup backend
cd backend
pip install -r requirements.txt
python app.py

# Open frontend in browser
open frontend/index.html
