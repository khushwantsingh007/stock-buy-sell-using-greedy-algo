from flask import Flask, render_template, jsonify, send_from_directory
import yfinance as yf
import datetime
import numpy as np
import pandas as pd
import threading
import time
import os
from threading import Lock

app = Flask(__name__, static_folder='static', static_url_path='/static')

# Config
LOOKBACK_DAYS = 365  # 1 year for better indicator calculation
REFRESH_INTERVAL = 60  # seconds

class TradingAlgorithm:
    def __init__(self):
        self.buy_threshold = 30  # RSI below this is oversold (buy)
        self.sell_threshold = 70  # RSI above this is overbought (sell)
        self.sma_short = 20
        self.sma_long = 50

    def generate_signals(self, df):
        """Generate buy/sell signals based on technical indicators"""
        try:
            signals = pd.DataFrame(index=df.index)
            signals['Price'] = df['Close']
            signals['Signal'] = 'HOLD'
            
            # RSI-based signals
            signals.loc[df['RSI'] < self.buy_threshold, 'Signal'] = 'BUY'
            signals.loc[df['RSI'] > self.sell_threshold, 'Signal'] = 'SELL'
            
            # SMA crossover signals - only override if RSI is neutral
            neutral_mask = (df['RSI'] >= self.buy_threshold) & (df['RSI'] <= self.sell_threshold)
            signals.loc[neutral_mask & (df['SMA_20'] > df['SMA_50']), 'Signal'] = 'BUY'
            signals.loc[neutral_mask & (df['SMA_20'] < df['SMA_50']), 'Signal'] = 'SELL'
            
            # Ensure signals have proper state management
            positions = ['HOLD']
            
            for i in range(1, len(signals)):
                current_signal = signals.iloc[i]['Signal']
                prev_position = positions[-1]
                
                if current_signal == 'BUY' and prev_position == 'BUY':
                    positions.append('BUY')
                elif current_signal == 'SELL' and prev_position == 'SELL':
                    positions.append('SELL')
                elif current_signal == 'HOLD':
                    positions.append(prev_position)
                else:
                    positions.append(current_signal)
            
            signals['Signal'] = positions
            
            return signals
        except Exception as e:
            print(f"Error generating signals: {str(e)}")
            return None

    def analyze_performance(self, signals):
        """Calculate the performance metrics based on trading signals"""
        try:
            if signals is None or signals.empty:
                return self._empty_performance_result()
                
            positions = []
            current_position = None
            profit = 0
            
            for i in range(1, len(signals)):
                prev_signal = signals.iloc[i-1]['Signal']
                current_signal = signals.iloc[i]['Signal']
                price = signals.iloc[i]['Price']
                
                if prev_signal != 'BUY' and current_signal == 'BUY':
                    positions.append({
                        'type': 'BUY',
                        'date': signals.index[i],
                        'price': price
                    })
                    current_position = positions[-1]
                
                elif prev_signal != 'SELL' and current_signal == 'SELL' and current_position is not None:
                    trade_profit = price - current_position['price']
                    profit += trade_profit
                    positions.append({
                        'type': 'SELL',
                        'date': signals.index[i],
                        'price': price,
                        'profit': trade_profit
                    })
                    current_position = None
            
            profit_trades = [p for p in positions if p.get('type') == 'SELL' and p.get('profit', 0) > 0]
            all_sell_trades = [p for p in positions if p.get('type') == 'SELL']
            
            win_rate = 0
            if all_sell_trades:
                win_rate = len(profit_trades) / len(all_sell_trades)
            
            return {
                'total_trades': len(all_sell_trades),
                'profitable_trades': len(profit_trades),
                'win_rate': round(win_rate * 100, 2),
                'total_profit': round(profit, 2),
                'positions': positions[-5:] if positions else []
            }
        except Exception as e:
            print(f"Error analyzing performance: {str(e)}")
            return self._empty_performance_result()
    
    def _empty_performance_result(self):
        return {
            'total_trades': 0,
            'profitable_trades': 0,
            'win_rate': 0,
            'total_profit': 0,
            'positions': []
        }


class StockManager:
    def __init__(self):
        self.cache = {}
        self.last_updated = {}
        self.cache_lock = Lock()
        self.trading_algo = TradingAlgorithm()
        self.exchange_suffixes = {
            'NSE': '.NS',
            'BSE': '.BO'
        }

    def normalize_ticker(self, ticker):
        """Normalize ticker by adding exchange suffix if needed"""
        if not ticker:
            return None
            
        ticker = ticker.upper()
        
        if any(ticker.endswith(suffix) for suffix in self.exchange_suffixes.values()):
            return ticker
            
        if ':' in ticker:
            exchange, symbol = ticker.split(':')
            if exchange in self.exchange_suffixes:
                return symbol + self.exchange_suffixes[exchange]
                
        return ticker + '.NS'

    def fetch_data(self, ticker):
        """Fetch stock data from YFinance and calculate indicators"""
        if not ticker or not isinstance(ticker, str) or len(ticker) > 15:
            return None
            
        try:
            normalized_ticker = self.normalize_ticker(ticker)
            end_date = datetime.datetime.today().strftime('%Y-%m-%d')
            start_date = (datetime.datetime.today() - datetime.timedelta(days=LOOKBACK_DAYS)).strftime('%Y-%m-%d')
            
            df = yf.download(normalized_ticker, start=start_date, end=end_date, progress=False)
            if df.empty:
                print(f"No data found for {normalized_ticker}")
                return None
                
            if len(df) < 50:
                print(f"Not enough data points for {normalized_ticker} (only {len(df)} rows)")
                return None
                
            df = df.dropna()
            df = self.add_indicators(df)
            
            with self.cache_lock:
                self.cache[ticker] = df
                self.last_updated[ticker] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
            return df
        except Exception as e:
            print(f"Error fetching {ticker}: {str(e)}")
            return None

    def add_indicators(self, df):
        """Add technical indicators to the dataframe"""
        try:
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss.replace(0, np.finfo(float).eps)
            df['RSI'] = 100 - (100 / (1 + rs))
            
            return df
        except Exception as e:
            print(f"Error adding indicators: {str(e)}")
            return df

    def generate_signals(self, ticker):
        """Generate trading signals for a given ticker"""
        try:
            with self.cache_lock:
                if ticker not in self.cache:
                    df = self.fetch_data(ticker)
                    if df is None:
                        return None
                else:
                    df = self.cache[ticker]
            
            signals = self.trading_algo.generate_signals(df)
            if signals is None:
                return None
                
            last_signal = signals.iloc[-1]['Signal']
            last_date = signals.index[-1].strftime('%Y-%m-%d')
            
            signals_data = []
            for i, row in signals.reset_index().iterrows():
                signals_data.append({
                    'Date': row['Date'].strftime('%Y-%m-%d'),
                    'Price': float(row['Price']),
                    'Signal': row['Signal']
                })
            
            return {
                'signals': signals_data,
                'recommendation': last_signal,
                'recommendation_date': last_date,
                'analysis': self.trading_algo.analyze_performance(signals)
            }
        except Exception as e:
            print(f"Error generating signals for {ticker}: {str(e)}")
            return None


stock_manager = StockManager()

def background_updater():
    """Background thread to periodically update stock data"""
    while True:
        try:
            with stock_manager.cache_lock:
                tickers = list(stock_manager.cache.keys())
                
            for ticker in tickers:
                stock_manager.fetch_data(ticker)
                
            time.sleep(REFRESH_INTERVAL)
        except Exception as e:
            print(f"Background updater error: {str(e)}")
            time.sleep(5)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                           'favicon.ico', mimetype='image/vnd.microsoft.icon')


@app.route('/api/stock/<ticker>')
def stock_api(ticker):
    """API endpoint to get stock data and analysis"""
    try:
        if not ticker or not isinstance(ticker, str) or len(ticker) > 15:
            return jsonify({'error': 'Invalid ticker symbol'}), 400

        if ticker not in stock_manager.cache:
            df = stock_manager.fetch_data(ticker)
            if df is None:
                return jsonify({'error': f'Could not fetch data for {ticker}'}), 404

        signals = stock_manager.generate_signals(ticker)
        if signals is None:
            return jsonify({'error': f'Could not generate signals for {ticker}'}), 500
        
        with stock_manager.cache_lock:
            df = stock_manager.cache[ticker]
            last_updated = stock_manager.last_updated.get(ticker, 'Never')
            
        prices_data = []
        for date, row in df.iterrows():
            prices_data.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Open': float(row['Open']),
                'High': float(row['High']),
                'Low': float(row['Low']),
                'Close': float(row['Close']),
                'Volume': int(row['Volume'])
            })
        
        return jsonify({
            'prices': prices_data,
            'indicators': {
                'SMA_20': df['SMA_20'].fillna(0).tolist(),
                'SMA_50': df['SMA_50'].fillna(0).tolist(),
                'RSI': df['RSI'].fillna(0).tolist()
            },
            'signals': signals,
            'last_updated': last_updated
        })
    except Exception as e:
        print(f"API error for {ticker}: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500


if __name__ == '__main__':
    os.makedirs('static/js', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    
    threading.Thread(target=background_updater, daemon=True).start()
    
    app.run(debug=True, use_reloader=False)