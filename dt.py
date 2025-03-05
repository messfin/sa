import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np

# Set page config
st.set_page_config(page_title="Advanced Stock Analysis", layout="wide")

def calculate_ema(data, period):
    return data['Close'].ewm(span=period, adjust=False).mean()

def calculate_vwap(df):
    # Reset index to handle date
    df = df.copy()  # Create a copy to avoid modifying original
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    volume_price = typical_price * df['Volume']
    cumulative_volume = df['Volume'].cumsum()
    cumulative_volume_price = volume_price.cumsum()
    return cumulative_volume_price / cumulative_volume

def calculate_macd(data):
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def calculate_rsi(data, periods=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def generate_signals(df):
    """Generate buy and sell signals"""
    df = df.copy()
    
    # MACD Line crosses above Signal Line (Buy)
    df['Buy_Signal'] = ((df['MACD'] > df['Signal']) & 
                        (df['MACD'].shift(1) <= df['Signal'].shift(1)) & 
                        (df['RSI'] < 70)).astype(int)
    
    # MACD Line crosses below Signal Line (Sell)
    df['Sell_Signal'] = ((df['MACD'] < df['Signal']) & 
                         (df['MACD'].shift(1) >= df['Signal'].shift(1)) & 
                         (df['RSI'] > 30)).astype(int)
    
    return df

# Sidebar inputs
st.sidebar.header('User Input Parameters')
ticker = st.sidebar.text_input("Stock Symbol", "AAPL")
today = datetime.today()
default_start = today - timedelta(days=365)
start_date = st.sidebar.date_input("Start Date", default_start)
end_date = st.sidebar.date_input("End Date", today)

@st.cache_data
def load_stock_data(symbol, start, end):
    try:
        df = yf.download(symbol, start=start, end=end)
        if df.empty:
            st.error("No data found for the selected stock and date range")
            return None
        
        # Calculate technical indicators
        df['EMA9'] = calculate_ema(df, 9)
        df['EMA20'] = calculate_ema(df, 20)
        df['VWAP'] = calculate_vwap(df)
        df['MACD'], df['Signal'] = calculate_macd(df)
        df['RSI'] = calculate_rsi(df)
        
        # Generate buy/sell signals
        df = generate_signals(df)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.write("Debug info:", e.__class__.__name__)
        return None

# Main content
st.title('Advanced Stock Analysis Dashboard')

# Load data
df = load_stock_data(ticker, start_date, end_date)

if df is not None:
    # Create subplots
    fig = make_subplots(rows=3, cols=1, 
                        shared_xaxes=True,
                        vertical_spacing=0.05,
                        row_heights=[0.6, 0.2, 0.2])

    # Main price chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price',
        increasing_line_color='green',
        decreasing_line_color='red'
    ), row=1, col=1)

    # Add EMAs
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['EMA9'],
        name='9 EMA',
        line=dict(color='blue', width=1)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['EMA20'],
        name='20 EMA',
        line=dict(color='orange', width=1)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['VWAP'],
        name='VWAP',
        line=dict(color='purple', width=1)
    ), row=1, col=1)

    # Add buy signals
    buy_mask = df['Buy_Signal'] == 1
    if buy_mask.any():
        buy_signals = df[buy_mask]
        fig.add_trace(go.Scatter(
            x=buy_signals.index,
            y=buy_signals['Low'] * 0.99,
            mode='markers+text',
            marker=dict(
                symbol='triangle-up',
                size=15,
                color='green',
                line=dict(width=2, color='darkgreen')
            ),
            text='BUY',
            textposition='bottom center',
            textfont=dict(size=12, color='green'),
            name='Buy Signal'
        ), row=1, col=1)

    # Add sell signals
    sell_mask = df['Sell_Signal'] == 1
    if sell_mask.any():
        sell_signals = df[sell_mask]
        fig.add_trace(go.Scatter(
            x=sell_signals.index,
            y=sell_signals['High'] * 1.01,
            mode='markers+text',
            marker=dict(
                symbol='triangle-down',
                size=15,
                color='red',
                line=dict(width=2, color='darkred')
            ),
            text='SELL',
            textposition='top center',
            textfont=dict(size=12, color='red'),
            name='Sell Signal'
        ), row=1, col=1)

    # Update the main chart y-axis
    fig.update_yaxes(title_text="Price", row=1, col=1)

    # Update layout
    fig.update_layout(
        title=f'{ticker} Technical Analysis',
        yaxis_title="Price",
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    # Make sure candlesticks are visible
    fig.update_layout(
        yaxis=dict(
            autorange=True,
            fixedrange=False
        )
    )

    # MACD subplot with signals
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MACD'],
        name='MACD',
        line=dict(color='blue', width=1)
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Signal'],
        name='Signal',
        line=dict(color='orange', width=1)
    ), row=2, col=1)

    # Add MACD histogram
    colors = ['red' if val < 0 else 'green' for val in df['MACD'] - df['Signal']]
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['MACD'] - df['Signal'],
        marker_color=colors,
        name='MACD Histogram'
    ), row=2, col=1)

    # Add buy signals on MACD
    buy_mask = df['Buy_Signal'] == 1
    if buy_mask.any():
        buy_signals = df[buy_mask]
        fig.add_trace(go.Scatter(
            x=buy_signals.index,
            y=buy_signals['MACD'],
            mode='markers+text',
            marker=dict(
                symbol='triangle-up',
                size=12,
                color='green',
                line=dict(width=2, color='darkgreen')
            ),
            text='BUY',
            textposition='bottom center',
            textfont=dict(size=10, color='green'),
            name='MACD Buy',
            showlegend=False
        ), row=2, col=1)

    # Add sell signals on MACD
    sell_mask = df['Sell_Signal'] == 1
    if sell_mask.any():
        sell_signals = df[sell_mask]
        fig.add_trace(go.Scatter(
            x=sell_signals.index,
            y=sell_signals['MACD'],
            mode='markers+text',
            marker=dict(
                symbol='triangle-down',
                size=12,
                color='red',
                line=dict(width=2, color='darkred')
            ),
            text='SELL',
            textposition='top center',
            textfont=dict(size=10, color='red'),
            name='MACD Sell',
            showlegend=False
        ), row=2, col=1)

    # Add annotations for crossovers
    for idx, row in buy_signals.iterrows():
        fig.add_annotation(
            x=idx,
            y=row['MACD'],
            text='↑',
            showarrow=False,
            font=dict(size=14, color='green'),
            row=2, col=1
        )

    for idx, row in sell_signals.iterrows():
        fig.add_annotation(
            x=idx,
            y=row['MACD'],
            text='↓',
            showarrow=False,
            font=dict(size=14, color='red'),
            row=2, col=1
        )

    # RSI subplot with signals
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['RSI'],
        name='RSI',
        line=dict(color='purple', width=1)
    ), row=3, col=1)

    # Add RSI buy signals
    if buy_mask.any():
        fig.add_trace(go.Scatter(
            x=buy_signals.index,
            y=buy_signals['RSI'],
            mode='markers',
            marker=dict(
                symbol='triangle-up',
                size=8,
                color='green',
                line=dict(width=1, color='darkgreen')
            ),
            name='RSI Buy',
            showlegend=False
        ), row=3, col=1)

    # Add RSI sell signals
    if sell_mask.any():
        fig.add_trace(go.Scatter(
            x=sell_signals.index,
            y=sell_signals['RSI'],
            mode='markers',
            marker=dict(
                symbol='triangle-down',
                size=8,
                color='red',
                line=dict(width=1, color='darkred')
            ),
            name='RSI Sell',
            showlegend=False
        ), row=3, col=1)

    # Add RSI levels
    fig.add_shape(
        type='line',
        x0=df.index[0],
        x1=df.index[-1],
        y0=70,
        y1=70,
        line=dict(color='red', width=1, dash='dash'),
        row=3,
        col=1
    )

    fig.add_shape(
        type='line',
        x0=df.index[0],
        x1=df.index[-1],
        y0=30,
        y1=30,
        line=dict(color='green', width=1, dash='dash'),
        row=3,
        col=1
    )

    # Update RSI axis range
    fig.update_yaxes(range=[0, 100], row=3, col=1)

    # Update layout
    fig.update_layout(
        title=f'{ticker} Technical Analysis',
        xaxis_title="Date",
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    # Update y-axes grid lines
    fig.update_yaxes(
        gridcolor='lightgrey',
        gridwidth=0.1,
        zerolinecolor='lightgrey',
        zerolinewidth=1
    )

    # Update x-axes grid lines
    fig.update_xaxes(
        gridcolor='lightgrey',
        gridwidth=0.1,
        zerolinecolor='lightgrey',
        zerolinewidth=1
    )

    # Display the plot
    st.plotly_chart(fig, use_container_width=True)

    # Statistics and Analysis
    st.subheader('Technical Indicators Summary')
    col1, col2, col3 = st.columns(3)
    
    with col1:
        current_rsi = float(df['RSI'].iloc[-1])
        st.metric("RSI", f"{current_rsi:.2f}", 
                 "Overbought > 70, Oversold < 30")
    
    with col2:
        current_macd = float(df['MACD'].iloc[-1])
        current_signal = float(df['Signal'].iloc[-1])
        macd_signal = "Bullish" if current_macd > current_signal else "Bearish"
        st.metric("MACD Signal", macd_signal)
    
    with col3:
        current_close = float(df['Close'].iloc[-1])
        current_ema = float(df['EMA20'].iloc[-1])
        trend = "Bullish" if current_close > current_ema else "Bearish"
        st.metric("Trend (20 EMA)", trend)

    # Export data option
    if st.button('Export Data to CSV'):
        csv = df.to_csv()
        st.download_button(
            label="Download Data",
            data=csv,
            file_name=f'{ticker}_technical_analysis.csv',
            mime='text/csv'
        )
else:
    st.error("No data available for the selected stock symbol and date range.")