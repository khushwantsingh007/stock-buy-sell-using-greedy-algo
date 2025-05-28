import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from enum import Enum

class Position(Enum):
    LONG = "LONG"
    SHORT = "SHORT" 
    CASH = "CASH"

class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class EnhancedTradingAlgorithm:
    def __init__(self, 
                 rsi_buy_threshold: float = 30,
                 rsi_sell_threshold: float = 70,
                 sma_short: int = 20,
                 sma_long: int = 50,
                 volume_threshold: float = 1.2,  # Volume spike multiplier
                 stop_loss_pct: float = 0.05,   # 5% stop loss
                 take_profit_pct: float = 0.10): # 10% take profit
        
        self.rsi_buy_threshold = rsi_buy_threshold
        self.rsi_sell_threshold = rsi_sell_threshold
        self.sma_short = sma_short
        self.sma_long = sma_long
        self.volume_threshold = volume_threshold
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # Strategy weights
        self.rsi_weight = 0.25
        self.sma_weight = 0.25
        self.volume_weight = 0.15
        self.momentum_weight = 0.10
        self.bollinger_weight = 0.15
        self.macd_weight = 0.10

    def add_enhanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators"""
        try:
            if df.empty or len(df) < 50:
                return df
                
            # Basic moving averages
            df['SMA_20'] = df['Close'].rolling(window=self.sma_short, min_periods=self.sma_short).mean()
            df['SMA_50'] = df['Close'].rolling(window=self.sma_long, min_periods=self.sma_long).mean()
            
            # RSI calculation with proper handling
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
            
            # Avoid division by zero
            loss = loss.replace(0, np.finfo(float).eps)
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['BB_Middle'] = df['Close'].rolling(window=20, min_periods=20).mean()
            bb_std = df['Close'].rolling(window=20, min_periods=20).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
            
            # BB Width and Position
            bb_width = df['BB_Upper'] - df['BB_Lower']
            df['BB_Width'] = bb_width / df['BB_Middle'].replace(0, np.finfo(float).eps)
            df['BB_Position'] = (df['Close'] - df['BB_Lower']) / bb_width.replace(0, np.finfo(float).eps)
            
            # MACD
            exp1 = df['Close'].ewm(span=12, min_periods=12).mean()
            exp2 = df['Close'].ewm(span=26, min_periods=26).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_Signal'] = df['MACD'].ewm(span=9, min_periods=9).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            
            # Volume indicators
            df['Volume_SMA'] = df['Volume'].rolling(window=20, min_periods=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA'].replace(0, np.finfo(float).eps)
            
            # Price momentum
            df['Price_Change_5d'] = df['Close'].pct_change(periods=5)
            df['Price_Momentum'] = df['Close'].rolling(window=10, min_periods=10).apply(
                lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if len(x) >= 10 and x.iloc[0] != 0 else 0, 
                raw=False
            )
            
            # Volatility
            close_mean = df['Close'].rolling(window=20, min_periods=20).mean()
            close_std = df['Close'].rolling(window=20, min_periods=20).std()
            df['Volatility'] = close_std / close_mean.replace(0, np.finfo(float).eps)
            
            return df
            
        except Exception as e:
            print(f"Error adding enhanced indicators: {str(e)}")
            return df

    def calculate_signal_strength(self, row: pd.Series) -> Tuple[float, Dict[str, float]]:
        """Calculate signal strength from multiple indicators"""
        signals = {}
        
        # RSI Signal (-1 to 1)
        if pd.notna(row['RSI']):
            if row['RSI'] <= self.rsi_buy_threshold:
                signals['rsi'] = 1.0  # Strong buy
            elif row['RSI'] >= self.rsi_sell_threshold:
                signals['rsi'] = -1.0  # Strong sell
            else:
                # Gradual scaling between thresholds
                rsi_normalized = (row['RSI'] - 50) / 20  # Scale around 50
                signals['rsi'] = -rsi_normalized  # Invert: low RSI = buy signal
        else:
            signals['rsi'] = 0.0
            
        # SMA Signal (-1 to 1)
        if pd.notna(row['SMA_20']) and pd.notna(row['SMA_50']) and row['SMA_50'] > 0:
            sma_diff = (row['SMA_20'] - row['SMA_50']) / row['SMA_50']
            signals['sma'] = np.tanh(sma_diff * 10)  # Sigmoid scaling
        else:
            signals['sma'] = 0.0
            
        # Volume Signal (0 to 1 for buy, 0 to -1 for sell)
        if pd.notna(row['Volume_Ratio']):
            if row['Volume_Ratio'] > self.volume_threshold:
                # High volume supports the trend
                price_trend = signals.get('sma', 0) + signals.get('rsi', 0)
                signals['volume'] = 0.5 * np.sign(price_trend) if price_trend != 0 else 0
            else:
                signals['volume'] = 0.0
        else:
            signals['volume'] = 0.0
            
        # Momentum Signal (-1 to 1)
        if pd.notna(row['Price_Momentum']):
            signals['momentum'] = np.tanh(row['Price_Momentum'] * 5)
        else:
            signals['momentum'] = 0.0
            
        # Bollinger Bands Signal
        if pd.notna(row['BB_Position']):
            if row['BB_Position'] <= 0.1:  # Near lower band
                signals['bollinger'] = 0.8
            elif row['BB_Position'] >= 0.9:  # Near upper band
                signals['bollinger'] = -0.8
            else:
                signals['bollinger'] = 0.0
        else:
            signals['bollinger'] = 0.0
            
        # MACD Signal
        if pd.notna(row['MACD']) and pd.notna(row['MACD_Signal']):
            macd_diff = row['MACD'] - row['MACD_Signal']
            signals['macd'] = np.tanh(macd_diff * 10)
        else:
            signals['macd'] = 0.0
        
        # Weighted combination
        total_signal = (
            signals['rsi'] * self.rsi_weight +
            signals['sma'] * self.sma_weight +
            signals['volume'] * self.volume_weight +
            signals['momentum'] * self.momentum_weight +
            signals['bollinger'] * self.bollinger_weight +
            signals['macd'] * self.macd_weight
        )
        
        # Clamp between -1 and 1
        total_signal = max(-1.0, min(1.0, total_signal))
        
        return total_signal, signals

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate enhanced buy/sell signals with proper position management"""
        try:
            if df.empty or len(df) < max(self.sma_long, 50):
                return pd.DataFrame()
                
            # Add enhanced indicators
            df = self.add_enhanced_indicators(df)
            
            signals_df = pd.DataFrame(index=df.index)
            signals_df['Price'] = df['Close']
            signals_df['Signal'] = SignalType.HOLD.value
            signals_df['Signal_Strength'] = 0.0
            signals_df['Position'] = Position.CASH.value
            
            # Calculate signal strength for each row
            signal_strengths = []
            signal_details = []
            
            for idx, row in df.iterrows():
                strength, details = self.calculate_signal_strength(row)
                signal_strengths.append(strength)
                signal_details.append(details)
                
            signals_df['Signal_Strength'] = signal_strengths
            
            # Generate position-aware signals
            current_position = Position.CASH
            entry_price = None
            buy_threshold = 0.3
            sell_threshold = -0.3
            
            for i in range(len(signals_df)):
                current_price = signals_df.iloc[i]['Price']
                signal_strength = signals_df.iloc[i]['Signal_Strength']
                
                # Determine signal based on strength thresholds
                if signal_strength > buy_threshold:  # Buy threshold
                    if current_position == Position.CASH:
                        signals_df.iloc[i, signals_df.columns.get_loc('Signal')] = SignalType.BUY.value
                        current_position = Position.LONG
                        entry_price = current_price
                    else:
                        signals_df.iloc[i, signals_df.columns.get_loc('Signal')] = SignalType.HOLD.value
                        
                elif signal_strength < sell_threshold:  # Sell threshold
                    if current_position == Position.LONG:
                        signals_df.iloc[i, signals_df.columns.get_loc('Signal')] = SignalType.SELL.value
                        current_position = Position.CASH
                        entry_price = None
                    else:
                        signals_df.iloc[i, signals_df.columns.get_loc('Signal')] = SignalType.HOLD.value
                        
                else:  # Hold or risk management
                    if current_position == Position.LONG and entry_price and entry_price > 0:
                        # Check stop loss and take profit
                        price_change = (current_price - entry_price) / entry_price
                        
                        if price_change <= -self.stop_loss_pct:  # Stop loss
                            signals_df.iloc[i, signals_df.columns.get_loc('Signal')] = SignalType.SELL.value
                            current_position = Position.CASH
                            entry_price = None
                        elif price_change >= self.take_profit_pct:  # Take profit
                            signals_df.iloc[i, signals_df.columns.get_loc('Signal')] = SignalType.SELL.value
                            current_position = Position.CASH
                            entry_price = None
                        else:
                            signals_df.iloc[i, signals_df.columns.get_loc('Signal')] = SignalType.HOLD.value
                    else:
                        signals_df.iloc[i, signals_df.columns.get_loc('Signal')] = SignalType.HOLD.value
                
                signals_df.iloc[i, signals_df.columns.get_loc('Position')] = current_position.value
            
            return signals_df
            
        except Exception as e:
            print(f"Error generating enhanced signals: {str(e)}")
            return pd.DataFrame()

    def analyze_performance(self, signals: pd.DataFrame) -> Dict:
        """Enhanced performance analysis with risk metrics"""
        try:
            if signals.empty:
                return self._empty_performance_result()
                
            trades = []
            current_trade = None
            
            for i, row in signals.iterrows():
                if row['Signal'] == SignalType.BUY.value:
                    if current_trade is None:  # Open new position
                        current_trade = {
                            'entry_date': i,
                            'entry_price': row['Price'],
                            'type': 'LONG'
                        }
                        trades.append({
                            'type': 'BUY',
                            'date': i,
                            'price': row['Price']
                        })
                        
                elif row['Signal'] == SignalType.SELL.value:
                    if current_trade is not None:  # Close position
                        profit = row['Price'] - current_trade['entry_price']
                        profit_pct = profit / current_trade['entry_price'] if current_trade['entry_price'] > 0 else 0
                        
                        trades.append({
                            'type': 'SELL',
                            'date': i,
                            'price': row['Price'],
                            'profit': profit,
                            'profit_pct': profit_pct,
                            'days_held': (i - current_trade['entry_date']).days
                        })
                        current_trade = None
            
            # Calculate performance metrics
            sell_trades = [t for t in trades if t['type'] == 'SELL']
            
            if not sell_trades:
                return self._empty_performance_result()
                
            profits = [t['profit'] for t in sell_trades]
            profit_pcts = [t['profit_pct'] for t in sell_trades]
            
            total_profit = sum(profits)
            profitable_trades = len([p for p in profits if p > 0])
            win_rate = profitable_trades / len(sell_trades) if sell_trades else 0
            
            # Risk metrics
            avg_profit_pct = np.mean(profit_pcts) if profit_pcts else 0
            volatility = np.std(profit_pcts) if len(profit_pcts) > 1 else 0
            sharpe_ratio = avg_profit_pct / volatility if volatility > 0 else 0
            
            max_drawdown = self._calculate_max_drawdown(profits)
            avg_holding_period = np.mean([t['days_held'] for t in sell_trades]) if sell_trades else 0
            
            return {
                'total_trades': len(sell_trades),
                'profitable_trades': profitable_trades,
                'win_rate': round(win_rate * 100, 2),
                'total_profit': round(total_profit, 2),
                'avg_profit_pct': round(avg_profit_pct * 100, 2),
                'volatility': round(volatility * 100, 2),
                'sharpe_ratio': round(sharpe_ratio, 3),
                'max_drawdown': round(max_drawdown * 100, 2),
                'avg_holding_days': round(avg_holding_period, 1),
                'positions': trades[-10:] if trades else []  # Last 10 trades
            }
            
        except Exception as e:
            print(f"Error in performance analysis: {str(e)}")
            return self._empty_performance_result()

    def _calculate_max_drawdown(self, profits: List[float]) -> float:
        """Calculate maximum drawdown from profits"""
        if not profits:
            return 0.0
            
        cumulative = np.cumsum(profits)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / np.maximum(np.abs(running_max), 1)  # Avoid division by zero
        return abs(np.min(drawdown)) if len(drawdown) > 0 else 0.0

    def _empty_performance_result(self) -> Dict:
        """Return empty performance metrics"""
        return {
            'total_trades': 0,
            'profitable_trades': 0,
            'win_rate': 0,
            'total_profit': 0,
            'avg_profit_pct': 0,
            'volatility': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'avg_holding_days': 0,
            'positions': []
        }

    def get_current_recommendation(self, df: pd.DataFrame) -> Dict:
        """Get current recommendation with detailed reasoning"""
        try:
            if df.empty:
                return {
                    'recommendation': 'HOLD', 
                    'confidence': 0, 
                    'signal_strength': 0.0,
                    'reasoning': 'No data available'
                }
                
            signals = self.generate_signals(df)
            if signals.empty:
                return {
                    'recommendation': 'HOLD', 
                    'confidence': 0, 
                    'signal_strength': 0.0,
                    'reasoning': 'Unable to generate signals'
                }
                
            latest = signals.iloc[-1]
            latest_data = df.iloc[-1]
            
            signal_strength = latest['Signal_Strength']
            recommendation = latest['Signal']
            
            # Calculate confidence (0-100)
            confidence = min(abs(signal_strength) * 100, 100)
            
            # Generate reasoning
            reasoning_parts = []
            
            if pd.notna(latest_data['RSI']):
                if latest_data['RSI'] < 30:
                    reasoning_parts.append(f"RSI at {latest_data['RSI']:.1f} indicates oversold conditions")
                elif latest_data['RSI'] > 70:
                    reasoning_parts.append(f"RSI at {latest_data['RSI']:.1f} indicates overbought conditions")
                else:
                    reasoning_parts.append(f"RSI at {latest_data['RSI']:.1f} is in neutral territory")
                    
            if pd.notna(latest_data['SMA_20']) and pd.notna(latest_data['SMA_50']):
                if latest_data['SMA_20'] > latest_data['SMA_50']:
                    reasoning_parts.append("Short-term trend is bullish (SMA20 > SMA50)")
                else:
                    reasoning_parts.append("Short-term trend is bearish (SMA20 < SMA50)")
                    
            if pd.notna(latest_data['Volume_Ratio']) and latest_data['Volume_Ratio'] > self.volume_threshold:
                reasoning_parts.append(f"High volume activity ({latest_data['Volume_Ratio']:.1f}x average)")
                
            if pd.notna(latest_data['MACD']) and pd.notna(latest_data['MACD_Signal']):
                if latest_data['MACD'] > latest_data['MACD_Signal']:
                    reasoning_parts.append("MACD shows bullish momentum")
                else:
                    reasoning_parts.append("MACD shows bearish momentum")
                    
            reasoning = '; '.join(reasoning_parts) if reasoning_parts else 'Mixed signals from technical indicators'
            
            return {
                'recommendation': recommendation,
                'confidence': round(confidence, 1),
                'signal_strength': round(signal_strength, 3),
                'reasoning': reasoning
            }
            
        except Exception as e:
            print(f"Error getting recommendation: {str(e)}")
            return {
                'recommendation': 'HOLD', 
                'confidence': 0, 
                'signal_strength': 0.0,
                'reasoning': 'Error in analysis'
            }