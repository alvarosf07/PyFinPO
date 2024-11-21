# ``PyFinPO`` - Python Financial Portfolio Optimization
Welcome to PyFinPO, my personal library for Financial Portfolio Optimization in Python.  

PyFinPO is an abstraction of other Python libraries for Portfolio Optimization, aiming to provide a structured, simple and extensible infrastructure to perform Financial Portfolio Optimization.

### Context
When I first set out to develop a Portfolio Optimization Library in Python, my goal was simple: to gain a deep understanding of portfolio theory and its main methods. Building everything from scratch felt like the best way to learn the nuances of the different models, approaches, estimators... and their implementations.

However, as I started exploring the existing Portfolio Optimization libraries in Python, I discovered that there are already VERY GOOD resources available for free ([see list below](#source-libraries)). I quickly came to the realization that it made no sense to implement my own portfolio tool from scratch having all those open-source libraries available –tested and improved through user feedback– that already included most of the Portfolio Optimization tools needed. Reinventing the wheel no longer seemed practical or efficient.

So I changed my mind. Instead of creating yet another standalone library, I decided that in order to understand the nuances of Portfolio Optimization, it would make more sense for me to build a centralized library which organizes the different functionalities available in other open-source libraries under a single, cohesive API. By integrating their strengths with an intuitive high-level interface, this tool could help me (and potentially other users) to seamlessly access the best functionalities each library has to offer without needing to master multiple APIs.

Additionally, I decided to prioritize modularity and extensibility, making extremely easy to add new proprietary models and integrate them in the buit-in API with minimal effort.

In a nutshell, I defined 3 main principles for the project: unified structure, intuitive use, and scalable modularity.


### Objectives

PyFinPO aims to satisfy 3 main objectives:

1. To unify the different Portfolio Optimization functionalities offered in other libraries under a common, structured API.
2. To provide a high-level interface to intuitively perform Portfolio Optimization with Python.
3. To offer a MODULAR and EXTENSIBLE tool which anyone can use to build proprietary models and feed them into the created API structure to leverage the already implemented functionalities in existing libraries.


### Note
```
Please note that PyFinPO Library is still under construction, therefore many of the functionalities have not been implemented yet. The summary tables below detail all the necessary information about the functionalities, documentation links and source code status for each model.
```


</br>

# Table of Contents

1) [Installation](#1-installation)
2) [Library Structure](#2-library-structure)
3) [Features](#3-features)
    - 3.0) [Portfolio Selection Problem Overview]()
    - 3.1) [Input Estimates](#31-input-estimators)
        - 3.1.1) [Expected Return Models](#311-expected-return-models)
        - 3.1.2) [Risk Models](#312-risk-models)
    - 3.2) [Portfolio Optimization](#32-portfolio-optimization-po)
        - 3.2.1) [Portfolio Optimization Models](#321-portfolio-optimization-models)
4) [Source Libraries](#4-source-libraries)
5) [Future Works](#5-future-works)

</br>

# 1) Installation
If you would like to install ``PyFinPO``, you can try any of the following 3 options:

 1. Clone the project and use the source code:

 ```bash
 git clone https://github.com/alvarosf07/pypot
 ```

 2. Alternatively, you can just install the project directly from your terminal:

 ```bash
 pip install -e git+https://github.com/alvarosf07/pypot.git
 ```
 
 3. However, it is best practice to use a dependency manager within a virtual environment. You can import all the dependencies within ``PyFinPO`` to your local environment by cloning/downloading the project and then in the project directory just run:

 ```bash
 python setup.py install
 ```

# 2) Library Structure

# 3) Features

## 3.0) Portfolio Selection Problem Overview

## 3.1) Input Estimators

### 3.1.1) Expected Return Models

#### Historical Averages
| Model Tag         | Model Name                                           | Documentation                                                               | Implementation                                                                  |
|-------------------|----------------------------------------------------- |---------------------------------------------------------------------------- |---------------------------------------------------------------------  |
| ``mh_rm``         | Mean Historical Return Model  (aka Empirical Returns)                       | [PyPortOpt](https://pyportfolioopt.readthedocs.io/en/latest/ExpectedReturns.html#pypfopt.expected_returns.mean_historical_return)  | [mh_rm.py](./pyfpo/input_estimates/expected_return_models/mh_rm.py)   |
| ``ewmh_rm``       | Exponentially Weighted Mean Return Model             | [PyPortOpt](https://pyportfolioopt.readthedocs.io/en/latest/ExpectedReturns.html#pypfopt.expected_returns.ema_historical_return)   | [ewmh_rm.py](./pyfpo/input_estimates/expected_return_models/ewmh_rm.py)  |
| ``mceq_rm``       | Market Cap Equilibrium Expected Return Model             | [skfolio](https://skfolio.org/generated/skfolio.moments.EquilibriumMu.html)   | *To be implemented*  |
| ``eqweq_rm``       | Equal Weight Equilibrium Expected Return Model             | [skfolio](https://skfolio.org/generated/skfolio.moments.EquilibriumMu.html)   | *To be implemented*  |


#### Economic & Factor Models
| Model Tag         | Model Name                                               | Documentation                                                               | Implementation                                                                  |
|-------------------|--------------------------------------------------------- |---------------------------------------------------------------------------- |---------------------------------------------------------------------  |
| ``capm_rm``       | Capital Asset Pricing Return Model (Equilibrium)         | [PyPortOpt](https://pyportfolioopt.readthedocs.io/en/latest/ExpectedReturns.html#pypfopt.expected_returns.capm_return)   | [capm_rm.py](./pyfpo/input_estimates/expected_return_models/capm_rm.py) |
| ``apt_rm``        | Arbitrage Pricing Theory (APT) Return Model              | [See docs](https://www.fe.training/free-resources/portfolio-management/arbitrage-pricing-theory/)   | _To be implemented_ |
| ``fama_factor_rm``| Fama-French Three-Factor Return Model                    | [See docs](https://en.wikipedia.org/wiki/Fama%E2%80%93French_three-factor_model)   | _To be implemented_ |
| ``multi_factor_rm``| Multi-Factor Return Models        | [skfolio](https://skfolio.org/auto_examples/1_mean_risk/plot_13_factor_model.html#factor-model)   | _To be implemented_ | 



#### Statistical/Machine Learning Models
- **Shrinkage Estimators** -> The estimator shrinks the sample mean toward a target vector:
- **Bayesian Models** -> Model relationships between variables and incorporate new information/views to estimate probabilistic returns.
- **Time-Series Forecasting Return Models**
- **Regression-Based Return Models** -> Predict future returns using historical and fundamental data
- **Supervised ML Models** -> Support Vector Machines, Gradient Boosting, Random Forests...
- **Neural Networks** -> Capture non-linear relationships and patterns, suitable for complex datasets.


| Model Tag              | Model Name                                               | Documentation                                                          | Implementation                                                                      |
|------------------------|--------------------------------------------------------- |------------------------------------------------------------------------|---------------------------------------------------------------------------|
| | | | |
| **Shrinkage Estimators**  | | | |
| ``shrinkage_rm``  | Shrinkage Return Models Estimators                       | [skfolio - Expected Return Shrinkage](https://skfolio.org/generated/skfolio.moments.ShrunkMu.html#skfolio.moments.ShrunkMu)   | _To be implemented_ |
| ``shrinkage_rm.james_stein``  | Shrinkage Return Models - James-Stein        | [skfolio - Expected Return Shrinkage](https://skfolio.org/generated/skfolio.moments.ShrunkMu.html#skfolio.moments.ShrunkMu)   | _To be implemented_ |
| ``shrinkage_rm.bayes_stein``  | Shrinkage Return Models - Bayes-Stein        | [skfolio - Expected Return Shrinkage](https://skfolio.org/generated/skfolio.moments.ShrunkMu.html#skfolio.moments.ShrunkMu)  | _To be implemented_ |
| ``shrinkage_rm.bop``  | Shrinkage Return Models - Bodnar Okhrin Parolya      | [skfolio - Expected Return Shrinkage](https://skfolio.org/generated/skfolio.moments.ShrunkMu.html#skfolio.moments.ShrunkMu)  | _To be implemented_ |
| **Bayesian Models**  | | | |
| ``black_litterman_rm``       |  Black Litterman Model (return + risk model)        | [PyPortOpt - Bkack-Litterman](https://pyportfolioopt.readthedocs.io/en/latest/BlackLitterman.html)  | [black_litterman_rm.py](./pyfpo/input_estimates/robust_models/black_litterman_rm.py) |
<!-- | ``TBD``       |  -        | [Bayesian Predictive Return Distributions](https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp969.pdf)   | _To be implemented_ |
| **Regression-Based Return Models**  | | | |
| ``TBD``       |          | []()   | _To be implemented_ |
| **Time-Series Forecasting Return Models**  | | | |
| ``TBD``       |          | []()   | _To be implemented_ |
| **Supervised ML Models**  | | | |
| ``TBD``       |          | []()   | _To be implemented_ |
| **Neural Networks**  | | | |
| ``TBD``       |         | []()   | _To be implemented_ | -->



#### Hybrid Models
Combine any of the previous models for more robust estimates.
| Model Tag         | Model Name                                               | Documentation                                                               | Code                                                                  |
|-------------------|--------------------------------------------------------- |---------------------------------------------------------------------------- |---------------------------------------------------------------------  |
| ``TBD``           |  -         | []()   | _To be implemented_ |


</br>

### 3.1.2) Risk Models

#### Covariance Estimators
| Model Tag              | Model Name                                               | Documentation                                                          | Code                                                                      |
|------------------------|--------------------------------------------------------- |------------------------------------------------------------------------|---------------------------------------------------------------------------|
| ``sample_cov``         | Sample Covariance Risk Model                             | [RiskModels.md](./docs/input_estimates/RiskModels.md)                  | [sample_cov.py](./pyfpo/input_estimates/risk_models/sample_cov.py)        |
| ``empirical_cov``      | Empirical Covariance (Max Likelihood Covariance Estimator)  | [Scikit-learn](https://scikit-learn.org/1.5/modules/covariance.html#empirical-covariance)                     | _To be implemented_             |
| ``implied_cov``        | Implied Covariance Risk Model                            | [Implied Covariance Matrix](https://users.ugent.be/~yrosseel/lavaan/evermann_slides.pdf)                         | _To be implemented_             |
| ``semi_cov``           | Semi-covariance Risk Model   | [Semi-Covariance Matrix](https://random-docs.readthedocs.io/en/latest/portfolio_optimisation/risk_estimators.html#semi-covariance-matrix)    | [semi_cov.py](./pyfpo/input_estimates/risk_models/semi_cov.py)  |
| ``ew_cov``             | Exponentially-Weigthed Covariance Risk Model             | [EW Covariance Matrix](https://random-docs.readthedocs.io/en/latest/portfolio_optimisation/risk_estimators.html#exponentially-weighted-covariance-matrix)                  | [ew_cov.py](./pyfpo/input_estimates/risk_models/ew_cov.py)                |
| ``cov_denoising``      | Covariance Denoising Risk Model                          | [Covariance Denoising](https://random-docs.readthedocs.io/en/latest/portfolio_optimisation/risk_estimators.html#de-noising-and-de-toning-covariance-correlation-matrix)  | _To be implemented_                |
| ``cov_detoning``       | Covariance Detoning Risk Model                          | [Covariance Detoning](https://random-docs.readthedocs.io/en/latest/portfolio_optimisation/risk_estimators.html#de-toning)  | _To be implemented_             |


#### Covariance Shrinkage
| Model Tag              | Model Name                                               | Documentation                                                          | Code                                                                      |
|------------------------|--------------------------------------------------------- |------------------------------------------------------------------------|---------------------------------------------------------------------------|
| ``cov_shrinkage``      | Covariance Shrinkage Risk Models                         | [RiskModels.md](./docs/input_estimates/RiskModels.md)                  | [cov_shrinkage.py](./pyfpo/input_estimates/risk_models/cov_shrinkage.py)  |
| ``cov_shrinkage.shrunk_covariance``    | Covariance Shrinkage - Manual Shrinkage     | [RiskModels.md](./docs/input_estimates/RiskModels.md)               | [cov_shrinkage.py](./pyfpo/input_estimates/risk_models/cov_shrinkage.py)  |
| ``cov_shrinkage.ledoit_wolf``          | Covariance Shrinkage - Ledoit-Wolf          | [RiskModels.md](./docs/input_estimates/RiskModels.md)               | [cov_shrinkage.py](./pyfpo/input_estimates/risk_models/cov_shrinkage.py)  |
| ``cov_shrinkage.oracle_approximating`` | Covariance Shrinkage - Oracle Approximating | [RiskModels.md](./docs/input_estimates/RiskModels.md)               | [cov_shrinkage.py](./pyfpo/input_estimates/risk_models/cov_shrinkage.py)  |


#### Sparse Inverse Covariance Estimators
| Model Tag              | Model Name                                               | Documentation                                                          | Code                                                                      |
|------------------------|--------------------------------------------------------- |------------------------------------------------------------------------|---------------------------------------------------------------------------|
| ``graph_lasso_cov``    | Sparse Inverse Graphical Lasso Covariance Estimator         | [Scikit-learn](https://scikit-learn.org/1.5/modules/covariance.html#sparse-inverse-covariance)          | _To be implemented_                |


#### Robust Covariance Estimators
| Model Tag              | Model Name                                               | Documentation                                                          | Code                                                                      |
|------------------------|--------------------------------------------------------- |------------------------------------------------------------------------|---------------------------------------------------------------------------|
| ``mcd_cov``            | Robust Minimum Covariance Determinant (MCD) Estimator    | [Scikit-learn](https://scikit-learn.org/1.5/modules/covariance.html#robust-covariance-estimation)                | _To be implemented_             |
| ``gerber_cov``         | Robust Gerber Statistic for Covariance Estimation        | [The Gerber Statistic](https://portfoliooptimizer.io/blog/the-gerber-statistic-a-robust-co-movement-measure-for-correlation-matrix-estimation/)    | _To be implemented_ |

</br>

## 3.2) Portfolio Optimization (PO)

### 3.2.1) Portfolio Optimization Models

#### Naive PO Models
| Model Tag              | Model Name                                               | Documentation                                                          | Code                                                                      |
|------------------------|--------------------------------------------------------- |------------------------------------------------------------------------|---------------------------------------------------------------------------|
| ``eqw_po``             | Equal Weight (1/N) Portfolio Optimization                | [skfolio](https://skfolio.org/generated/skfolio.optimization.EqualWeighted.html#skfolio.optimization.EqualWeighted)           | _To be implemented_             |
| ``ivp_po``             | Inverse Volatility Portfolio (IVP) Optimization          | [skfolio](https://skfolio.org/generated/skfolio.optimization.InverseVolatility.html#skfolio.optimization.InverseVolatility)   | _To be implemented_             |
| ``random_po``          | Random Portfolio Optimization (Dirichlet Distribution)   | [skfolio](https://skfolio.org/generated/skfolio.optimization.Random.html#skfolio.optimization.Random)                         | _To be implemented_             |

#### Risk-Based PO Models
| Model Tag              | Model Name                                               | Documentation                                                          | Code                                                                      |
|------------------------|--------------------------------------------------------- |------------------------------------------------------------------------|---------------------------------------------------------------------------|
| ``gmv_po``*        | Global Minimum Variance PO                       | [PyPortOpt](https://pyportfolioopt.readthedocs.io/en/latest/MeanVariance.html#pypfopt.efficient_frontier.EfficientFrontier.min_volatility), [GMVP](http://showcase2.imw.tuwien.ac.at/BWOpt/PF1_minvar.html) | _To be implemented_    |
| ``maxdiv_po``*     | Maximum Diversification Portfolio Optimization   | [skfolio](https://skfolio.org/generated/skfolio.optimization.MaximumDiversification.html#skfolio.optimization.MaximumDiversification) | _To be implemented_    |
| ``risk_parity_po``     | Risk Parity Portfolio Optimization                       | [skfolio](https://skfolio.org/generated/skfolio.optimization.RiskBudgeting.html), [Risk Parity PO](https://tspace.library.utoronto.ca/bitstream/1807/106376/4/Costa_Del_Pozo_Giorgio_202106_PhD_thesis.pdf) | _To be implemented_    |
| ``risk_budgeting_po``  | Risk Budgeting Portfolio Optimization                    | [skfolio](https://skfolio.org/generated/skfolio.optimization.RiskBudgeting.html), [Risk Budgeting PO](http://www.columbia.edu/~mh2078/A_generalized_risk_budgeting_approach.pdf)                            | _To be implemented_    |

- **Both ``gmv_po`` and ``maxdiv_po`` are not implemented as strict PO models, but as objective functions for optimization inside other PO models (such as Mean-Risk Models detailed below).*

#### Mean-Risk PO Models
Mean-Risk Portfolio Optimization models aim to find asset combinations which optimize the relationship return vs risk. The most well-known Mean-Risk model is Mean-Variance Portfolio Optimization, which uses the variance of asset returns as measure of risk. However, many different measures of risk can be selected, giving rise to a wide range of Mean-Risk models. 

``PyFinPO`` provides direct implementation of the most relevant Mean-Risk models, as detailed in the summary table below. In order to select any other measure of risk which is not directly implemented in the PO models below, the parent class ``MeanRiskPO`` generalizes all mean-risk models and allows to choose among any of the available risk metrics (see list below).

| Model Tag              | Model Name                                               | Documentation                                                          | Code                                                                      |
|------------------------|--------------------------------------------------------- |------------------------------------------------------------------------|---------------------------------------------------------------------------|
| ``mr_po``              | Mean-Risk Portfolio Optimization (generalizes all models below)                                | [skfolio](https://skfolio.org/generated/skfolio.optimization.MeanRisk.html)  | _To be implemented_             |
| ``mv_po``              | Mean-Variance Portfolio Optimization                             | [PyPortOpt](https://pyportfolioopt.readthedocs.io/en/latest/MeanVariance.html)  | [mv_po.py](./pyfinpo/portfolio_optimization/po_models/mv_po.py)             |
| ``msv_po``             | Mean-Semivariance Portfolio Optimization                         | [PyPortOpt](https://pyportfolioopt.readthedocs.io/en/latest/GeneralEfficientFrontier.html#efficient-semivariance)  | [msv_po.py](./pyfinpo/portfolio_optimization/po_models/msv_po.py)             |
| ``mcvar_po``           | Mean-CVaR (Conditional Value at Risk) Portfolio Optimization     | [PyPortOpt](https://pyportfolioopt.readthedocs.io/en/latest/GeneralEfficientFrontier.html#efficient-cvar)  | [mcvar_po.py](./pyfinpo/portfolio_optimization/po_models/mcvar_po.py)             |
| ``mcdar_po``           | Mean-CDaR (Conditional Drawdown at Risk) Portfolio Optimization  | [PyPortOpt](https://pyportfolioopt.readthedocs.io/en/latest/GeneralEfficientFrontier.html#efficientcdar)  | [mcdar_po.py](./pyfinpo/portfolio_optimization/po_models/mcdar_po.py)             |

*List of Risk Measures Supported under ``MeanRiskPO`` class*:
- Mean Absolute Deviation
- Variance (Second Central Moment)
- <del> Skewness (Normalized Third Central Moment)
- <del> Kurtosis (Normalized Fourth Central Moment minus 3)
- <del> Fourth Central Moment
- First Lower Partial Moment 
- Semi-Variance (Second Lower Partial Moment)
- <del> Fourth Lower Partial Moment
- <del> Value at Risk
- Maximum Drawdown
- Average Drawdown
- <del> Drawdown at Risk
- <del> Entropic Risk Measure
- CVaR (Conditional Value at Risk)
- EVaR (Entropic Value at Risk)
- CDaR (Conditional Drawdown at Risk)
- EDaR (Entropic Drawdown at Risk)
- Worst Realization
- Ulcer Index
- Gini Mean Difference


#### Robust Mean-Risk PO Models
| Model Tag              | Model Name                                               | Documentation                                                          | Code                                                                      |
|------------------------|--------------------------------------------------------- |------------------------------------------------------------------------|---------------------------------------------------------------------------|
| ``cla_po``             | Critical Line Algorithm (CLA) Portfolio Optimization     | [PyPortOpt](https://pyportfolioopt.readthedocs.io/en/latest/OtherOptimizers.html#the-critical-line-algorithm)    | [cla_po.py](./pyfpo/portfolio_optimization/po_models/cla_po.py)    |
| ``dr_cvar_po``         | Distributionally Robust CVaR Portfolio Optimization      | [skfolio](https://skfolio.org/auto_examples/4_distributionally_robust_cvar/plot_1_distributionally_robust_cvar.html#distributionally-robust-cvar)   | *To be implemented* |


#### Clustering PO Models
| Model Tag              | Model Name                                               | Documentation                                                          | Code                                                                      |
|------------------------|--------------------------------------------------------- |------------------------------------------------------------------------|---------------------------------------------------------------------------|
| ``hrp_po``             | Hierarchical Risk Parity (HRP) Portfolio Optimization    | [mlfinlab](https://random-docs.readthedocs.io/en/latest/portfolio_optimisation/hierarchical_risk_parity.html)  | [hrp_po.py](./pyfpo/portfolio_optimization/po_models/hrp_po.py)    |
| ``herc_po``            | Hierarchical Equal Risk Contribution (HERC) Portfolio Optimization | [mlfinlab](https://random-docs.readthedocs.io/en/latest/portfolio_optimisation/hierarchical_equal_risk_contribution.html) | _To be implemented_    |
| ``nco_po``             | Nested Clustered Optimization (NCO)                      | [mlfinlab](https://random-docs.readthedocs.io/en/latest/portfolio_optimisation/nested_clustered_optimisation.html)        | _To be implemented_    |

#### Ensemble PO Models
| Model Tag              | Model Name                                               | Documentation                                                          | Code                                                                      |
|------------------------|--------------------------------------------------------- |------------------------------------------------------------------------|---------------------------------------------------------------------------|
| ``stack_po``           | Stacking Portfolio Optimization                          |   -                                              | _To be implemented_                                                       |

</br>

# 4) Source Libraries
- [PyPortOpt](https://github.com/robertmartin8/PyPortfolioOpt)
- [Riskfolio-Lib](https://github.com/dcajasn/Riskfolio-Lib)
- [skfolio](https://github.com/skfolio/skfolio)
- [scikit-portfolio](https://github.com/scikit-portfolio/scikit-portfolio)
- [cardiel](https://github.com/thk3421-models/cardiel)
- []() 
- []()

</br>

# 5) Future Works



<!--
skfolio missing pieces: 

Distance Estimator:
Pearson Distance
Kendall Distance
Spearman Distance
Covariance Distance (based on any of the above covariance estimators)
Distance Correlation
Variation of Information

Prior Estimator:
Empirical
Black & Litterman
Factor Model

Uncertainty Set Estimator:
On Expected Returns:
Empirical
Circular Bootstrap

On Covariance:
Empirical
Circular bootstrap

Pre-Selection Transformer:
Non-Dominated Selection
Select K Extremes (Best or Worst)
Drop Highly Correlated Assets

Cross-Validation and Model Selection:
Compatible with all sklearn methods (KFold, etc.)
Walk Forward
Combinatorial Purged Cross-Validation

Hyper-Parameter Tuning:
Compatible with all sklearn methods (GridSearchCV, RandomizedSearchCV)

Optimization Features:
Minimize Risk
Maximize Returns
Maximize Utility
Maximize Ratio
Transaction Costs
Management Fees
L1 and L2 Regularization
Weight Constraints
Group Constraints
Budget Constraints
Tracking Error Constraints
Turnover Constraints
-->







<!--
riskfolio-lib missing pieces: 




-->









<!--
scikit--portfolio missing pieces: 




-->