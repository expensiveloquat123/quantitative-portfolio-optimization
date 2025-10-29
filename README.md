# Quantitative Portfolio Optimization

A professional Python toolkit for portfolio optimization using Modern Portfolio Theory. Implements mean-variance optimization, efficient frontier analysis, and comprehensive risk metrics with position constraints.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

This project demonstrates quantitative finance concepts through a complete portfolio optimization framework. Built as part of my Master's studies in Business Engineering (Risk & Finance), this toolkit showcases:

- **Mean-variance optimization** with realistic constraints
- **Efficient frontier** calculation and visualization
- **Risk metrics** (Sharpe ratio, volatility, VaR)
- **Professional visualizations** for portfolio analysis

## Key Features

**Portfolio Optimization**
- Maximum Sharpe ratio portfolio
- Minimum volatility portfolio  
- Target return optimization
- Position limits (max 40% per asset to enforce diversification)

**Risk Analysis**
- Annual return and volatility calculations
- Sharpe ratio computation
- Risk-return tradeoff visualization

**Professional Visualizations**
- Efficient frontier with 5,000 random portfolios
- Correlation matrix heatmap
- Portfolio weight allocations
- Clean, publication-quality plots

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/expensivelouqat123/quantitative-portfolio-optimization.git
cd quantitative-portfolio-optimization
```

2. **Create virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Quick Start
```python
from src.data_loader import load_sample_data
from src.portfolio_optimizer import PortfolioOptimizer
from src.visualizations import PortfolioVisualizer

# Generate sample data
returns, _ = load_sample_data(num_assets=5, periods=252)

# Initialize optimizer
optimizer = PortfolioOptimizer(returns, risk_free_rate=0.02)

# Find optimal portfolios
max_sharpe = optimizer.optimize_max_sharpe(max_weight=0.40)
min_vol = optimizer.optimize_min_volatility(max_weight=0.40)

# Create visualizations
visualizer = PortfolioVisualizer()
efficient_frontier = optimizer.efficient_frontier()
visualizer.plot_efficient_frontier(efficient_frontier, max_sharpe, min_vol)
```

## Project Structure
```
quantitative-portfolio-optimization/
├── src/
│   ├── data_loader.py          # Data generation and preprocessing
│   ├── portfolio_optimizer.py  # Core optimization algorithms
│   ├── visualizations.py       # Plotting utilities
│   └── __init__.py
├── data/                        # Data directory
├── notebooks/                   # Jupyter notebooks (planned)
├── tests/                       # Unit tests (planned)
├── requirements.txt             # Python dependencies
└── README.md
```

## Mathematical Framework

This project implements **Modern Portfolio Theory (MPT)** as developed by Harry Markowitz:

### Mean-Variance Optimization

The optimizer solves:
```
minimize: w^T Σ w
subject to: w^T μ = μ_target
            w^T 1 = 1
            0 ≤ w_i ≤ 0.40  (position limit)
```

Where:
- `w` = vector of asset weights
- `Σ` = covariance matrix of returns
- `μ` = vector of expected returns

### Sharpe Ratio Maximization
```
maximize: (w^T μ - r_f) / √(w^T Σ w)
```

Where `r_f` is the risk-free rate (default: 2%).

## Example Output

### Maximum Sharpe Ratio Portfolio
```
Expected Return:  8.45%
Volatility:       15.23%
Sharpe Ratio:     0.423

Optimal Weights:
  Asset 1: 25.3%
  Asset 2: 18.7%
  Asset 3: 40.0%  (at constraint limit)
  Asset 4: 12.4%
  Asset 5:  3.6%
```

## Key Design Decisions

### Position Constraints
I implemented a **40% maximum weight per asset** constraint to:
- Enforce diversification (prevent concentration risk)
- Reflect real-world portfolio management practices
- Balance theoretical optimality with practical risk management

**Tradeoff:** The constrained portfolio achieves slightly lower Sharpe ratio than the unconstrained optimum, but provides better risk-adjusted returns through diversification.

### Sample Data vs. Real Data
Currently uses simulated data for demonstration. This approach:
- Ensures reproducibility
- Avoids external API dependencies
- Demonstrates the optimization methodology clearly

**Future enhancement:** Integration with real market data via APIs.

## Technical Implementation

- **Optimization:** SciPy's SLSQP algorithm for constrained optimization
- **Data Processing:** Pandas for time-series operations
- **Numerical Computing:** NumPy for matrix operations
- **Visualization:** Matplotlib and Seaborn for professional plots

## Limitations & Future Work

### Current Limitations
- Uses simulated data (not real market data)
- Assumes normally distributed returns
- Static optimization (single period)
- No transaction costs

### Planned Enhancements
- [ ] Integration with real market data (Yahoo Finance API)
- [ ] Backtesting framework
- [ ] Transaction cost modeling
- [ ] Multi-period optimization
- [ ] Risk parity strategies
- [ ] Black-Litterman model implementation

## Why This Project

As a Master's student in Business Engineering with focus on Risk & Finance, I'm particularly interested in **quantitative trading** and **systematic strategies**. This project demonstrates:

- Strong foundation in portfolio theory
- Practical Python implementation skills
- Understanding of optimization algorithms
- Ability to balance theory with real-world constraints

I'm actively seeking **quantitative trading/research internships** at firms like Optiver, IMC Trading, Jump Trading, and similar prop trading firms.

## Technologies Used

- **Python 3.12**
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **SciPy** - Optimization algorithms
- **Matplotlib/Seaborn** - Visualization
- **Scikit-learn** - Statistical utilities

## Learning Resources

This project was built using concepts from:
- Markowitz, H. (1952). "Portfolio Selection". *Journal of Finance*
- Sharpe, W. F. (1964). "Capital Asset Prices: A Theory of Market Equilibrium"
- Modern portfolio management best practices

## Contact

**Ana Lucia Leon**
- Master's Student - Business Engineering (Risk & Finance)
- GitHub: [@expensivelouqat123](https://github.com/expensivelouqat123)
- Interested in: Quantitative Finance, Algorithmic Trading, Risk Management

---

⭐ **Star this repo if you find it useful!**