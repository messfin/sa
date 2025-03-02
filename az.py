import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pytz
from pathlib import Path
import os
import numpy as np

def calculate_rsi(data, periods=14):
    """Calculate RSI for a given price series"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_eps_data(ticker_obj):
    """Get EPS data for the specified period"""
    try:
        # Get earnings data directly without date filtering
        earnings = ticker_obj.earnings_dates
        if earnings is not None and not earnings.empty:
            # Sort by date descending and get last 10 quarters
            earnings = earnings.sort_index(ascending=False).head(10)
            return earnings
    except Exception as e:
        print(f"Error fetching EPS data: {str(e)}")
    return None

def analyze_earnings_impact(ticker1, ticker2, days_before=5, days_after=5, output_dir="earnings_analysis"):
    """Analyze and compare stock performance around earnings dates"""
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Get stock data
        stock1 = yf.Ticker(ticker1)
        stock2 = yf.Ticker(ticker2)
        
        # Get EPS data first (last 10 quarters)
        eps1 = get_eps_data(stock1)
        eps2 = get_eps_data(stock2)
        
        if eps1 is None or eps2 is None or eps1.empty or eps2.empty:
            print(f"No earnings data found for {ticker1} or {ticker2}")
            return None
        
        # Calculate the date range based on earnings dates
        start_date = min(eps1.index.min(), eps2.index.min()) - timedelta(days=365)  # Extra year for MA calculation
        end_date = max(eps1.index.max(), eps2.index.max()) + timedelta(days=days_after)
        
        # Get historical price data
        df1 = stock1.history(start=start_date, end=end_date)
        df2 = stock2.history(start=start_date, end=end_date)
        
        # Calculate technical indicators
        for df in [df1, df2]:
            df['MA200'] = df['Close'].rolling(window=200).mean()
            df['RSI'] = calculate_rsi(df['Close'])
        
        # Export raw technical data
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        tech_data1 = output_dir / f'{ticker1}_technical_data_{timestamp}.csv'
        tech_data2 = output_dir / f'{ticker2}_technical_data_{timestamp}.csv'
        
        df1.to_csv(tech_data1)
        df2.to_csv(tech_data2)
        
        if df1.empty or df2.empty:
            print(f"No price data found for {ticker1} or {ticker2}")
            return None
            
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        return None

    def calculate_price_changes(df, earnings_dates, company_name):
        results = []
        
        for earning_date in earnings_dates.index:
            try:
                start_idx = earning_date - timedelta(days=days_before)
                end_idx = earning_date + timedelta(days=days_after)
                
                period_prices = df[start_idx:end_idx]
                
                if len(period_prices) < 2:
                    continue
                    
                pre_earnings_price = period_prices['Close'].iloc[0]
                post_earnings_price = period_prices['Close'].iloc[-1]
                pct_change = ((post_earnings_price - pre_earnings_price) / pre_earnings_price) * 100
                
                ma200 = period_prices['MA200'].iloc[0]
                rsi = period_prices['RSI'].iloc[0]
                
                # Get EPS data with safe fallbacks
                result_dict = {
                    'date': earning_date,
                    'pre_price': pre_earnings_price,
                    'post_price': post_earnings_price,
                    'pct_change': pct_change,
                    'company': company_name,
                    'volume': period_prices['Volume'].mean(),
                    'MA200': ma200,
                    'RSI': rsi,
                    'Above_MA200': pre_earnings_price > ma200,
                    'RSI_Level': 'Overbought' if rsi > 70 else 'Oversold' if rsi < 30 else 'Neutral'
                }
                
                # Try to add EPS data if available
                try:
                    result_dict.update({
                        'EPS_Actual': earnings_dates.loc[earning_date, 'EPS Actual'],
                        'EPS_Estimate': earnings_dates.loc[earning_date, 'EPS Estimate'],
                        'EPS_Surprise': earnings_dates.loc[earning_date, 'Surprise(%)']
                    })
                except:
                    result_dict.update({
                        'EPS_Actual': None,
                        'EPS_Estimate': None,
                        'EPS_Surprise': None
                    })
                
                results.append(result_dict)
                
            except Exception as e:
                print(f"Error processing earnings date {earning_date}: {str(e)}")
                continue
            
        return pd.DataFrame(results)
    
    # Calculate results for both companies
    results1 = calculate_price_changes(df1, eps1, ticker1)
    results2 = calculate_price_changes(df2, eps2, ticker2)
    
    if results1 is None or results2 is None or len(results1) == 0 or len(results2) == 0:
        print("Not enough data points to analyze")
        return None

    # Combine results
    all_results = pd.concat([results1, results2])
    
    # Create summary statistics
    summary_data = {
        'Metric': [
            'Average Change %',
            'Number of Earnings Events',
            'Positive Events',
            'Negative Events',
            'Average Volume',
            'Average RSI',
            'Events Above MA200 (%)',
            'RSI > 70 Events',
            'RSI < 30 Events'
        ],
        ticker1: [
            results1['pct_change'].mean(),
            len(results1),
            len(results1[results1['pct_change'] > 0]),
            len(results1[results1['pct_change'] < 0]),
            results1['volume'].mean(),
            results1['RSI'].mean(),
            (results1['Above_MA200'].sum() / len(results1)) * 100,
            len(results1[results1['RSI'] > 70]),
            len(results1[results1['RSI'] < 30])
        ],
        ticker2: [
            results2['pct_change'].mean(),
            len(results2),
            len(results2[results2['pct_change'] > 0]),
            len(results2[results2['pct_change'] < 0]),
            results2['volume'].mean(),
            results2['RSI'].mean(),
            (results2['Above_MA200'].sum() / len(results2)) * 100,
            len(results2[results2['RSI'] > 70]),
            len(results2[results2['RSI'] < 30])
        ]
    }

    # Add EPS metrics if available
    if 'EPS_Surprise' in results1.columns and 'EPS_Surprise' in results2.columns:
        eps_metrics = {
            'Metric': [
                'Average EPS Surprise %',
                'Positive EPS Surprises',
                'Negative EPS Surprises'
            ],
            ticker1: [
                results1['EPS_Surprise'].mean() if not results1['EPS_Surprise'].isna().all() else None,
                len(results1[results1['EPS_Surprise'] > 0]) if not results1['EPS_Surprise'].isna().all() else None,
                len(results1[results1['EPS_Surprise'] < 0]) if not results1['EPS_Surprise'].isna().all() else None
            ],
            ticker2: [
                results2['EPS_Surprise'].mean() if not results2['EPS_Surprise'].isna().all() else None,
                len(results2[results2['EPS_Surprise'] > 0]) if not results2['EPS_Surprise'].isna().all() else None,
                len(results2[results2['EPS_Surprise'] < 0]) if not results2['EPS_Surprise'].isna().all() else None
            ]
        }
        
        # Append EPS metrics to summary data
        for key in summary_data:
            summary_data[key].extend(eps_metrics[key])
    
    summary_df = pd.DataFrame(summary_data)
    
    # Create visualization
    if 'EPS_Surprise' in results1.columns and 'EPS_Surprise' in results2.columns and \
       not (results1['EPS_Surprise'].isna().all() and results2['EPS_Surprise'].isna().all()):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))
    
    # Plot 1: Price Changes
    companies = [ticker1, ticker2]
    avg_changes = [results1['pct_change'].mean(), results2['pct_change'].mean()]
    
    bars = ax1.bar(companies, avg_changes)
    ax1.set_title('Average Stock Price Change Around Earnings')
    ax1.set_ylabel('Average Percentage Change (%)')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    for bar, value in zip(bars, avg_changes):
        color = 'g' if value >= 0 else 'r'
        bar.set_color(color)
        ax1.text(
            bar.get_x() + bar.get_width()/2,
            value,
            f'{value:.1f}%',
            ha='center',
            va='bottom' if value > 0 else 'top'
        )
    
    # Plot 2: RSI Distribution
    ax2.boxplot([results1['RSI'], results2['RSI']], labels=companies)
    ax2.set_title('RSI Distribution at Earnings')
    ax2.set_ylabel('RSI Value')
    ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought')
    ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold')
    ax2.legend()
    
    # Plot 3: MA200 Position
    ma200_data = [
        (results1['Above_MA200'].sum() / len(results1)) * 100,
        (results2['Above_MA200'].sum() / len(results2)) * 100
    ]
    
    bars = ax3.bar(companies, ma200_data)
    ax3.set_title('Percentage of Earnings Events Above 200-day MA')
    ax3.set_ylabel('Percentage (%)')
    
    for bar, value in zip(bars, ma200_data):
        ax3.text(
            bar.get_x() + bar.get_width()/2,
            value,
            f'{value:.1f}%',
            ha='center',
            va='bottom'
        )
    
    # Plot 4: EPS Surprise (if data available)
    if 'EPS_Surprise' in results1.columns and 'EPS_Surprise' in results2.columns and \
       not (results1['EPS_Surprise'].isna().all() and results2['EPS_Surprise'].isna().all()):
        eps_surprises = [
            results1['EPS_Surprise'].mean() if not results1['EPS_Surprise'].isna().all() else 0,
            results2['EPS_Surprise'].mean() if not results2['EPS_Surprise'].isna().all() else 0
        ]
        bars = ax4.bar(companies, eps_surprises)
        ax4.set_title('Average EPS Surprise %')
        ax4.set_ylabel('Surprise Percentage')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        for bar, value in zip(bars, eps_surprises):
            if value != 0:
                color = 'g' if value >= 0 else 'r'
                bar.set_color(color)
                ax4.text(
                    bar.get_x() + bar.get_width()/2,
                    value,
                    f'{value:.1f}%',
                    ha='center',
                    va='bottom' if value > 0 else 'top'
                )
    
    plt.tight_layout()
    
    # Save outputs
    plot_filename = output_dir / f'technical_analysis_{ticker1}_{ticker2}.png'
    plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
    plt.close()
    
    # Save detailed results and summary
    detailed_csv = output_dir / f'detailed_results_{ticker1}_{ticker2}_{timestamp}.csv'
    summary_csv = output_dir / f'summary_{ticker1}_{ticker2}_{timestamp}.csv'
    
    all_results.to_csv(detailed_csv)
    summary_df.to_csv(summary_csv, index=False)
    
    print(f"\nAnalysis completed. Results exported to directory: {output_dir}")
    print(f"Files created:")
    print(f"1. Technical data for {ticker1}: {tech_data1}")
    print(f"2. Technical data for {ticker2}: {tech_data2}")
    print(f"3. Detailed results: {detailed_csv}")
    print(f"4. Summary statistics: {summary_csv}")
    print(f"5. Technical analysis plots: {plot_filename}")
    
    return all_results

def main():
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python script.py TICKER1 TICKER2")
        print("Example: python script.py GOOGL NVDA")
        return
        
    ticker1 = sys.argv[1].upper()
    ticker2 = sys.argv[2].upper()
    
    # Create output directory name using tickers and date
    output_dir = f"earnings_analysis_{ticker1}_{ticker2}_{datetime.now().strftime('%Y%m%d')}"
    
    print(f"Analyzing {ticker1} and {ticker2} earnings for the last 10 quarters")
    
    results = analyze_earnings_impact(
        ticker1,
        ticker2,
        days_before=5,
        days_after=5,
        output_dir=output_dir
    )

if __name__ == "__main__":
    main()