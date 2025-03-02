import unittest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch
import yfinance as yf
from earnings_sector_compare import StockAnalyzer  # Assuming your class is in this file


class TestStockAnalyzer(unittest.TestCase):

    @patch('yfinance.Ticker')
    def test_get_earnings_dates_success(self, mock_ticker):
        # Mock earnings data
        mock_earnings = pd.DataFrame(index=[datetime(2024, 1, 1), datetime(2023, 7, 1), datetime(2023, 4, 1)], data={'earnings': [1, 2, 3]})
        mock_ticker.return_value.earnings_dates = mock_earnings
        analyzer = StockAnalyzer()
        dates = analyzer.get_earnings_dates('AAPL')
        self.assertEqual(len(dates), 3)
        self.assertEqual(dates[0], datetime(2024, 1, 1))
        self.assertEqual(dates[1], datetime(2023, 7, 1))
        self.assertEqual(dates[2], datetime(2023, 4, 1))

    @patch('yfinance.Ticker')
    def test_get_earnings_dates_empty(self, mock_ticker):
        mock_ticker.return_value.earnings_dates = pd.DataFrame()
        analyzer = StockAnalyzer()
        dates = analyzer.get_earnings_dates('MSFT')
        self.assertEqual(len(dates), 0)

    @patch('yfinance.Ticker')
    def test_get_earnings_dates_error(self, mock_ticker):
        mock_ticker.side_effect = Exception("Mock yfinance error")
        analyzer = StockAnalyzer()
        dates = analyzer.get_earnings_dates('GOOG')
        self.assertEqual(len(dates), 0)

    @patch('yfinance.Ticker.history')
    def test_get_stock_data_success(self, mock_history):
        # Mock stock data
        mock_data = pd.DataFrame({'Close': [10, 12, 15, 14]}, index=pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04']))
        mock_history.return_value = mock_data
        analyzer = StockAnalyzer()
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 4)
        data = analyzer.get_stock_data('AAPL', start_date, end_date)
        self.assertIsNotNone(data)
        self.assertEqual(len(data), 4)

    @patch('yfinance.Ticker.history')
    def test_get_stock_data_empty(self, mock_history):
        mock_history.return_value = pd.DataFrame()
        analyzer = StockAnalyzer()
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 4)
        data = analyzer.get_stock_data('MSFT', start_date, end_date)
        self.assertIsNone(data)

    @patch('yfinance.Ticker.history')
    def test_get_stock_data_error(self, mock_history):
        mock_history.side_effect = Exception("Mock yfinance error")
        analyzer = StockAnalyzer()
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 4)
        data = analyzer.get_stock_data('GOOG', start_date, end_date)
        self.assertIsNone(data)

    @patch('yfinance.Ticker.history')
    def test_calculate_correlation_success(self, mock_history):
        # Mock data for correlation calculation
        mock_data1 = pd.Series([10, 12, 15, 14], name='Close1')
        mock_data2 = pd.Series([11, 13, 16, 13], name='Close2')
        analyzer = StockAnalyzer()
        correlation = analyzer.calculate_correlation(mock_data1, mock_data2)
        self.assertAlmostEqual(correlation, 1.0, places=2) # check if close to 1.0

    def test_calculate_correlation_insufficient_data(self):
        analyzer = StockAnalyzer()
        correlation = analyzer.calculate_correlation(pd.Series([1]), pd.Series([2]))
        self.assertIsNone(correlation)

    def test_get_correlation_category(self):
        analyzer = StockAnalyzer()
        self.assertEqual(analyzer.get_correlation_category(0.8), "High (80.00%)")
        self.assertEqual(analyzer.get_correlation_category(0.5), "Moderate (50.00%)")
        self.assertEqual(analyzer.get_correlation_category(0.2), "Low (20.00%)")
        self.assertEqual(analyzer.get_correlation_category(None), "N/A")


if __name__ == '__main__':
    unittest.main()