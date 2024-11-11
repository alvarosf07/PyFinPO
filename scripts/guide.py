import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
#from pypfopt.efficient_frontier import EfficientFrontier
#from pypfopt import risk_models
#from pypfopt import expected_returns

sys.path.append('/Users/alvarosanchez/Documents/Projects/personal-projects/pypot')
from pypo.input_estimates import expected_return_models

#df = pd.read_csv("data/raw/stock_prices_test.csv")

# 1) Get Data
tickers = ["ACN", "AMZN", "COST", "DIS", "F", "GILD", "JPM", "KO", "LUV", "MA", "MSFT", "PFE", "TSLA", "UNH", "XOM"]
ohlc = yf.download(tickers, period="max")
prices = ohlc["Adj Close"].dropna(how="all")

# 2) Inputs Estimation
# 2.1) Return Estimation
mu1 = expected_return_models.mh_rm(prices)
mu2 = expected_return_models.ewmh_rm(prices)
mu3 = expected_return_models.capm_rm(prices)
print (mu3)
# 2.2) Risk Estimation





# Mean-Variance Optimization - Pypfopt

# 1) Inputs:
#mu = expected_returns.mean_historical_return(df)
#S = risk_models.sample_cov(df)

# 2) Optimize for maximal Sharpe ratio
#ef = EfficientFrontier(mu, S)
#weights = ef.max_sharpe()
#ef.portfolio_performance(verbose=True)


# Mean-Variance Optimization - Personal Repo

# 1) Inputs:
expected_returns = expected_returns.mean_historical_return(df)
cov_matrix = risk_models.sample_cov(df)

# 2) Optimize for maximal Sharpe ratio
po = mv_po(expected_returns, cov_matrix)
pw = po.max_sharpe()
po.portfolio_performance(verbose=True)


