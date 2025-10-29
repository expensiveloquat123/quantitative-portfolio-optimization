import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Tuple, Optional, Dict
import warnings

warnings.filterwarnings('ignore')

class PortfolioOptimizer:
    "Mean variance Framework"
    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.02):
        """
        S1
        Args:
            returns: DataFrame of historical returns (rows=periods, col=assets)
            risk_free_rate: rf (default: 2% aka Historical Treasury Bills)
        """
        self.returns = returns
        self.mean_returns = returns.mean() * 252  # Annualized
        self.cov_matrix = returns.cov() * 252  # Annualized
        self.risk_free_rate = risk_free_rate
        self.num_assets = len(returns.columns)

    def portfolio_performance(self, weights: np.ndarray) -> Tuple[float, float]:
        """
        S2 (ER + volatility)
        Args: sum(weights) = 1 = 100%
        Out: (ER, volatility)
        """
        returns = np.sum(self.mean_returns * weights)
        std = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        return returns, std

    def negative_sharpe_ratio(self, weights: np.ndarray) -> float:
        """
        Args: array of weights
        Out: (-) sharpe ratio
        """
        returns, std = self.portfolio_performance(weights)
        sharpe = (returns - self.risk_free_rate) / std
        return -sharpe

    def portfolio_volatility(self, weights: np.ndarray) -> float:
        """
        Args: array of weights
        Out: Portfolio sd
        """
        return self.portfolio_performance(weights)[1]

    def optimize_max_sharpe(self,
                            long_only: bool = True,
                            max_weight: float = 1.0) -> Dict:
        """
        Must: find portfolio with max Sharpe ratio.
        Args:
            long_only: If True, restrict to long-only positions (weights >= 0)
            max_weight: Maximum weight for any single asset

        Out: dictionary with optimal weights, return, volatility, and Sharpe ratio
        """
        num_assets = self.num_assets

        # Initial guess: equal weights
        initial_weights = np.array([1.0 / num_assets] * num_assets)

        # sum(weights) = 1 = 100%
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

        # Bounds
        if long_only:
            bounds = tuple((0, max_weight) for _ in range(num_assets))
        else:
            bounds = tuple((-1, max_weight) for _ in range(num_assets))

        # Optimize
        result = minimize(
            self.negative_sharpe_ratio,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        opt_weights = result.x
        opt_return, opt_vol = self.portfolio_performance(opt_weights)
        opt_sharpe = (opt_return - self.risk_free_rate) / opt_vol

        return {
            'weights': opt_weights,
            'return': opt_return,
            'volatility': opt_vol,
            'sharpe_ratio': opt_sharpe
        }

    def optimize_min_volatility(self,
                                long_only: bool = True,
                                max_weight: float = 1.0) -> Dict:
        """
        Find minimum volatility portfolio.
        Args:
            long_only: If True, restrict to long-only positions
            max_weight: Maximum weight for any single asset
        Out: dictionary w/optimal weights, return, and volatility
        """
        num_assets = self.num_assets
        initial_weights = np.array([1.0 / num_assets] * num_assets)

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

        if long_only:
            bounds = tuple((0, max_weight) for _ in range(num_assets))
        else:
            bounds = tuple((-1, max_weight) for _ in range(num_assets))

        result = minimize(
            self.portfolio_volatility,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        opt_weights = result.x
        opt_return, opt_vol = self.portfolio_performance(opt_weights)

        return {
            'weights': opt_weights,
            'return': opt_return,
            'volatility': opt_vol,
            'sharpe_ratio': (opt_return - self.risk_free_rate) / opt_vol
        }

    def optimize_target_return(self,
                               target_return: float,
                               long_only: bool = True,
                               max_weight: float = 1.0) -> Optional[Dict]:
        """
        Find min(volatility) portfolio for a target return.
        Args:
            target_return: Desired annual return
            long_only: If True, restrict to long-only positions
            max_weight: Maximum weight for any single asset

        Out: dictionary w/optimal weights, return, and volatility (or None if infeasible)
        """
        num_assets = self.num_assets
        initial_weights = np.array([1.0 / num_assets] * num_assets)

        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: self.portfolio_performance(x)[0] - target_return}
        )

        if long_only:
            bounds = tuple((0, max_weight) for _ in range(num_assets))
        else:
            bounds = tuple((-1, max_weight) for _ in range(num_assets))

        result = minimize(
            self.portfolio_volatility,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if not result.success:
            return None

        opt_weights = result.x
        opt_return, opt_vol = self.portfolio_performance(opt_weights)

        return {
            'weights': opt_weights,
            'return': opt_return,
            'volatility': opt_vol,
            'sharpe_ratio': (opt_return - self.risk_free_rate) / opt_vol
        }

    def efficient_frontier(self,
                           num_portfolios: int = 100,
                           long_only: bool = True) -> pd.DataFrame:
        """
        Efficient frontier calc
        Args:
            num_portfolios: Number of portfolios to generate
            long_only: If True, restrict to long-only positions
        Out: DataFrame w/(returns, volatilities, and Sharpe ratios)
        """
        # Get min volatility and max Sharpe portfolios
        min_vol_port = self.optimize_min_volatility(long_only=long_only)
        max_sharpe_port = self.optimize_max_sharpe(long_only=long_only)

        # Define range of target returns
        min_return = min_vol_port['return']
        max_return = max_sharpe_port['return'] * 1.2  # Extend slightly beyond max Sharpe

        target_returns = np.linspace(min_return, max_return, num_portfolios)

        frontier_portfolios = []

        for target_ret in target_returns:
            port = self.optimize_target_return(target_ret, long_only=long_only)
            if port is not None:
                frontier_portfolios.append({
                    'Return': port['return'],
                    'Volatility': port['volatility'],
                    'Sharpe Ratio': port['sharpe_ratio']
                })

        return pd.DataFrame(frontier_portfolios)

    def random_portfolios(self, num_portfolios: int = 10000) -> pd.DataFrame:
        """
        Random portfolios to compare
        Args: num_portfolios = # of random portfolios to generate

        Out: DataFrame w/(returns, volatilities, and Sharpe ratios):
        """
        results = np.zeros((3, num_portfolios))

        for i in range(num_portfolios):
            # Generate random weights
            weights = np.random.random(self.num_assets)
            weights /= np.sum(weights)

            # Calculate performance
            port_return, port_vol = self.portfolio_performance(weights)
            sharpe = (port_return - self.risk_free_rate) / port_vol

            results[0, i] = port_return
            results[1, i] = port_vol
            results[2, i] = sharpe

        return pd.DataFrame({
            'Return': results[0],
            'Volatility': results[1],
            'Sharpe Ratio': results[2]
        })


# Example usage
if __name__ == "__main__":
    from data_loader import load_sample_data

    print("=" * 60)
    print("PORTFOLIO OPTIMIZATION DEMO")
    print("=" * 60)

    # Generate sample data
    returns, _ = load_sample_data(num_assets=5, periods=252)
    print(f"\nLoaded {len(returns)} days of returns for {len(returns.columns)} assets")

    # Initialize optimizer
    optimizer = PortfolioOptimizer(returns, risk_free_rate=0.02)

    # Find max Sharpe portfolio
    print("\n" + "=" * 60)
    print("MAXIMUM SHARPE RATIO PORTFOLIO")
    print("=" * 60)
    max_sharpe = optimizer.optimize_max_sharpe()
    print(f"Expected Return:  {max_sharpe['return']:.2%}")
    print(f"Volatility:       {max_sharpe['volatility']:.2%}")
    print(f"Sharpe Ratio:     {max_sharpe['sharpe_ratio']:.3f}")
    print("\nOptimal Weights:")
    for i, weight in enumerate(max_sharpe['weights']):
        print(f"  Asset {i + 1}: {weight:>7.2%}")

    # Find min volatility portfolio
    print("\n" + "=" * 60)
    print("MINIMUM VOLATILITY PORTFOLIO")
    print("=" * 60)
    min_vol = optimizer.optimize_min_volatility()
    print(f"Expected Return:  {min_vol['return']:.2%}")
    print(f"Volatility:       {min_vol['volatility']:.2%}")
    print(f"Sharpe Ratio:     {min_vol['sharpe_ratio']:.3f}")
    print("\nOptimal Weights:")
    for i, weight in enumerate(min_vol['weights']):
        print(f"  Asset {i + 1}: {weight:>7.2%}")

    print("\n" + "=" * 60)
    print("Optimization complete!")
    print("=" * 60)