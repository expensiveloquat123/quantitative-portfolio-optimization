"""
Data loading and preprocessing utilities for portfolio optimization.

This module handles fetching historical price data, calculating returns,
and preparing data for portfolio analysis.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Tuple, Optional
from datetime import datetime, timedelta
#import yfinance as yf
import yfinance as yf


class DataLoader:
    """
    Handles data loading and preprocessing for portfolio optimization.

    Attributes:
        tickers (List[str]): List of stock tickers
        start_date (str): Start date for historical data
        end_date (str): End date for historical data
    """

    def __init__(self, tickers: List[str], start_date: str, end_date: str):
        """
        Initialize DataLoader.

        Args:
            tickers: List of stock ticker symbols (e.g., ['AAPL', 'GOOGL', 'MSFT'])
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.prices = None
        self.returns = None

    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch historical adjusted close prices from Yahoo Finance.

        Returns:
            DataFrame with adjusted close prices for all tickers
        """
        print(f"Fetching data for {len(self.tickers)} tickers...")

        try:
            data = yf.download(
                self.tickers,
                start=self.start_date,
                end=self.end_date,
                progress=False
            )['Adj Close']

            # Handle single ticker case
            if len(self.tickers) == 1:
                data = data.to_frame()
                data.columns = self.tickers

            self.prices = data
            print(f"Successfully fetched {len(data)} days of data")
            return data

        except Exception as e:
            raise ValueError(f"Error fetching data: {str(e)}")

    def calculate_returns(self, method: str = 'simple') -> pd.DataFrame:
        """
        Calculate returns from price data.

        Args:
            method: 'simple' for arithmetic returns or 'log' for logarithmic returns

        Returns:
            DataFrame with calculated returns
        """
        if self.prices is None:
            raise ValueError("Must fetch data first using fetch_data()")

        if method == 'simple':
            self.returns = self.prices.pct_change().dropna()
        elif method == 'log':
            self.returns = np.log(self.prices / self.prices.shift(1)).dropna()
        else:
            raise ValueError("method must be 'simple' or 'log'")

        return self.returns

    def get_summary_statistics(self) -> pd.DataFrame:
        """
        Calculate summary statistics for returns.

        Returns:
            DataFrame with mean, std, min, max, and Sharpe ratio for each asset
        """
        if self.returns is None:
            raise ValueError("Must calculate returns first using calculate_returns()")

        stats = pd.DataFrame({
            'Mean': self.returns.mean() * 252,  # Annualized
            'Volatility': self.returns.std() * np.sqrt(252),  # Annualized
            'Min': self.returns.min(),
            'Max': self.returns.max(),
            'Sharpe': (self.returns.mean() * 252) / (self.returns.std() * np.sqrt(252))
        })

        return stats

    def save_data(self, returns_path: str = 'data/returns.csv',
                  prices_path: str = 'data/prices.csv'):
        """
        Save prices and returns to CSV files.

        Args:
            returns_path: Path to save returns data
            prices_path: Path to save price data
        """
        if self.prices is not None:
            self.prices.to_csv(prices_path)
            print(f"Prices saved to {prices_path}")

        if self.returns is not None:
            self.returns.to_csv(returns_path)
            print(f"Returns saved to {returns_path}")


def load_sample_data(num_assets: int = 5,
                     periods: int = 252) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generate sample return data for testing (when real data unavailable).

    Args:
        num_assets: Number of assets to simulate
        periods: Number of time periods

    Returns:
        Tuple of (returns DataFrame, correlation matrix)
    """
    np.random.seed(42)

    # Generate correlation matrix
    A = np.random.randn(num_assets, num_assets)
    corr_matrix = A @ A.T
    corr_matrix = corr_matrix / np.outer(np.sqrt(np.diag(corr_matrix)),
                                         np.sqrt(np.diag(corr_matrix)))

    # Generate returns
    means = np.random.uniform(0.05, 0.15, num_assets) / 252
    vols = np.random.uniform(0.15, 0.40, num_assets) / np.sqrt(252)

    returns = np.random.multivariate_normal(
        means,
        np.diag(vols) @ corr_matrix @ np.diag(vols),
        periods
    )

    dates = pd.date_range(end=datetime.now(), periods=periods, freq='D')
    tickers = [f'ASSET_{i + 1}' for i in range(num_assets)]

    returns_df = pd.DataFrame(returns, index=dates, columns=tickers)

    return returns_df, corr_matrix


# Example usage
if __name__ == "__main__":
    # Generate sample data (no internet needed!)
    print("Generating sample portfolio data...")
    sample_returns, corr = load_sample_data(num_assets=5, periods=252)

    print(f"\n✅ Generated {sample_returns.shape[0]} days of data for {sample_returns.shape[1]} assets")
    print("\nFirst few rows:")
    print(sample_returns.head())

    print("\nSummary Statistics:")
    print(sample_returns.describe())