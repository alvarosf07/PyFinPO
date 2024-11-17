# Intro to Portfolio Optimization
Portfolio Optimization (PO), technically known as Portfolio Selection Problem (PSP), is one of the leading problems in finance. Formally, the PSP can be stated as follows: given an amount of capital C and a set of n assets in which is possible to invest, how to assign now a fraction of the capital to each of the assets in such a way that, after a specified period of time T, some specific investment objectives are optimized.

The classical way of addressing the PSP has been the Mean-Variance (MV) model introduced by Markowitz back in 1952, where he formulated the problem as a trade-off between return and risk; considering the returns of individual securities as random variables and selecting their expected return (Mean) and Variance as measures of portfolio return and risk, respectively. However, despite the relevance of MV theory in portfolio selection, it presents several limitations that significantly impact both its practical and theoretical applications. For instance, it fails to account for skewness, kurtosis and other higher moments of the asset return distribution, it presents high sensitivity to input parameter estimations, and it only considers a single-period optimization. Moreover, MV theory does not bother well with certain constraints that make the portfolio selection problem not realistic in real trading environments, reducing its applicability in practice.

As consequence, many solutions have been proposed in research to improve the resolution and applicability of Portfolio Selection Problem. One of the main objectives of this library is to provide a base framework that serves as foundation on top of which we can build proprietary models that can improve the capabilities of Mean-Variance theory and related well-studied approaches to Portfolio Optimization.

For a complete introduction to porftolio optimization, including the main areas of the problem and references to other publications and sources that cover each of them in more detail, please see my [article publication](https://link.springer.com/chapter/10.1007/978-3-031-74186-9_21) [10].

</br>

# Problem Formulation
As the future performance of portfolio securities remains uncertain at the time of the investment decision, portfolio selection can be addressed as an opti- mization problem under uncertainty. In its most general form, the PSP can be formulated as a multi-objective mathematical optimization model, which can be written as follows according to [Zitzler’s formulation](https://ieeexplore.ieee.org/document/797969) [12]:

maximize: $\; \;f(x) = (f_1(x),f_2(x)...,f_p(x))$, 

subject to: $ \; \; c(x) = (c_1(x), c_2(x) . . . , c_m(x)) ≤ 0$, (1)

where: $ \; \; x = (w_1,w_2,...,w_n) ∈ X$.


The objective of the optimization process will be, based on a set of constraints $c(x)$ that define the problem, to determine the optimal decision variables $x = (w_1, w_2, . . . , w_n)$ –the portfolio weights– that maximize the objective function $f(x)$, which encompasses the investment objectives of choice.

A portfolio is thus represented by its weights $x = (w_1, w_2, . . . , w_n)$, where $w_i$ is the fraction of capital invested in asset $i$ $(i = 1, ..., n)$. To represent a portfolio, weights must satisfy a set of constraints that form a set X of feasible decision vectors. The feasible set $X$ can be defined as the set of decision vectors (set of possible portfolios) that satisfy all the constraints $c(x)$ of the problem [41]:

$X = \{ (w1,w2,...,wn) \; | \;c(x) ≤ 0, \; j = 1,...,m \} $

The presented PO formulation gives rise to a number of questions, which
lead to the proposed research areas of the PSP depicted in Figure 1:

1. Which are the parameters that should be selected as investment objectives to optimize, and how to choose between them?
2. How are the optimization parameters represented? (e.g. which return distri- bution is considered? Are parameters backward-looking or forward-looking? Are they treated as deterministic or considering estimation errors?)
3. How many objectives are considered for the optimization? Is the PO formu- lated as a single-objective problem, or as a multi-objective problem?
4. What is the time-horizon of the optimization? Once an investment decision is made, when and how to rebalance it?
5. What are the main constraints that should be considered to make the PSP applicable in real life? How do these constraints affect the PO?
6. What approach should be used to find a solution to the PO problem?

In the next section, an overview of these six topics is presented, detailing the
different approaches followed in literature to address each dimension of the PSP.

<br>

# Portfolio Selection Areas

<br>

## 1) Portfolio Selection Models

Portfolio Selection Models (PSM) refer to the different approaches used
to select the parameters to be optimized in the PSP, and the preference
relation (choice criterion) among them in the objective function. Throughout
literature, it's possible to distinguish three main PSMs
[9]: i) utility theory models, ii) parameter-based
models, and iii) stochastic dominance models.

Utility theory (UT) assumes that individuals possess a subjective
utility function that assigns different utility values for different
outcomes [9]. If the outcomes are determined by a
random variable (RV) like in the case of the PSP, then expected utility
theory (EUT) can be used as a rational criterion to order preference for
values of the RV. Several authors have proposed UT approaches to
represent the rational investing preferences of investors and, then,
solve the underlying optimization problem in which expected utility is
maximized [5; 9].

Parameter-based models (PBM) represent an alternative to EUT as instead
of developing a subjective utility function, they aim to characterize
portfolio performance by selecting a few parameters (statistics) that
describe the underlying portfolio return distributions
[9]. Depending on the parameters of choice, it is
possible to distinguish three main types of PBMs. The first and most
extended type are Mean-Risk (MR) models, which address the PSP in terms
of a trade-off between just two parameters: Mean (i.e. expected value of
portfolio return) and Risk. Markowitz [8] first
proposed variance as measure of risk, laying the foundations of
Mean-Variance (MV) theory. Subsequent MR models have proposed
alternative risk measures like non-symmetrical risk metrics (e.g.
semi-deviation, lower partial moments), as well as tail-risk metrics
such as Value at Risk (VaR), and Conditional Value at Risk (CVaR)
[9].

Alternative PBMs have tried to improve MR models either by adding a
third parameter to better characterize the distribution of portfolio
returns (e.g. CVaR [9], skewness [6]), or by
proposing alternative return-risk measures like Sharpe ratio
[1], risk-parity [5; @li2022multi] or omega
portfolios [7].

The last type of PSMs are stochastic dominance (SD) models, which
provide a sound theoretical basis for partial order of random variables,
helping determine whether one portfolio stochastically dominates another
in terms of risk-return characteristics [9]. Although
SD models are more theoretically attractive that utility theory and
PBMs, their practical applications are constrained due to the higher
computational complexity required.

<br>

## 2) Portfolio Parameters Estimation

Portfolio parameter estimation is concerned with the calculation of
inputs for the PSM of choice. This process is crucial in PO, as small
changes in estimates of input parameters can produce big differences in
optimization results [11].

There are three aspects of special importance in portfolio parameter
estimation. The first one is the choice of probability distribution to
model asset returns. Whereas MV theory is only concerned with mean and
variance, financial return distributions often exhibit skewness,
kurtosis and fat tails properties which cannot be characterized by their
mean-variances alone [11]. To capture such properties,
several authors have proposed modeling returns according to empirical,
fat tails or multi-variable distributions [5] in combination
with PSMs such as generalized expected utility, Mean-VaR or Mean-CVaR
models.

The second important aspect is predictive portfolio optimization (PPO).
Many studies in PO simply use past data as estimate of input parameters,
assumption which is not always accurate. As consequence, recent research
has focused on combining forecasting theory with portfolio optimization
in order to provide forward-looking estimates of input parameters
[1; @chen2021mean].

The third critical aspect in portfolio parameter estimation is the
consideration of estimation errors (EE). Typically, parameter
estimations are treated in a deterministic way (i.e. as certain,
error-free point estimates) without considering the uncertainty in the
estimation. With the objective of obtaining more realistic and robust
optimization results, several studies have proposed different approaches
that incorporate the uncertainty of parameters in the formulation of the
PO problem. As per [5], examples of these approaches include
the global minimum variance (GMV) portfolio, Bayesian techniques (such
as shrinkage approaches or the Black-Litterman model), robust
optimization techniques, or the inclusion of PO constraints that help
reduce estimation errors.

Other approaches also propose non-probabilistic models for future
expectations on PO input parameters through fuzzy theory or credibility
theory [11].

</br>

## 3) Portfolio Optimization Objectives

PO can be formulated as a single or multi-objective problem, depending
on the choice of number of investment objectives to optimize ($p$) in
Equation [\[Eq1\]].

The classical MV approach to the PSP is formulated as a single-objective
portfolio optimization (SOPO) [8], where the aim
is to minimize risk (variance) for a desired level of return --or
equivalently, maximize return for a given level of risk-- obtaining a
single optimal solution or efficient portfolio. Although SOPO has been
heavily used in research [4], it requires
knowing in advance either the return or the risk levels desired by the
investor, which is not always possible or sometimes simply not desirable
in real-world scenarios [2].

For this reason, an alternative approach which has gained attention over
the years is to address the PSP as a multi-objective portfolio
optimization (MOPO), which allows to provide a Pareto-optimal set of
solutions instead of a single optimal solution
[4]. There are two main methods to approach
MOPO. The most used is the weighted-sum method, which combines the
multiple objectives of the PO problem into a single one, assigning
weights to prioritize among them [2]. A second approach are
Pareto-based models, which use the concept of Pareto optimality
[3] to individually evaluate and rank the conflicting
objectives and identify the dominant set of portfolios.

</br>

## 4) Portfolio Optimization Period

The classical MVPO is formulated as a static single-period portfolio
optimization (SPPO), where a decision is made at the beginning of the
investment period and no further action is taken until the investment
horizon ends. However, this assumption is often not realistic in
real-world changing financial markets [5].

Two main approaches can be found in literature that address PO
dynamically, as a multi-period portfolio optimization (MPPO). The first
approach considers a discrete-time PO, where the expected utility of the
investor terminal wealth is maximized over a multi-period investment
horizon, and portfolio can be rebalanced only at discrete points in time
(e.g. for a 1-year PSP, adjusting portfolio weights at the beginning of
every month). The second approach is a continuous-time optimization,
where asset weights can be reallocated at any time within the investment
horizon. Zhang et. al. [11] provide a detailed review
on the formulation and advantages of dynamic PO techniques.

</br>

## 5) Portfolio Optimization Constraints

The original MV formulation of the PSP contains only one hard
constraint[^2], being that the sum of all asset weights must add up to
one (budget constraint). Nonetheless, portfolio management in practice
involves a lot of complexities and factors that are not properly
addressed by MV theory [5]. Consequently, the lack of
real-world constraints in theoretical PO models may often lead to
optimization results which cannot be reproduced in real life.

Table [1](#tab:PO_constraints) gathers the most common constraints that
are included in PO literature to make the PSP more applicable in
practice. These can be classified into three groups: the aforementioned
hard constraints, portfolio construction constraints, and problem
definition constraints. Portfolio construction constraints aim to
reproduce conditions and limitations faced when managing portfolios in
real-world trading environments, such as considering long and short
positions, transaction costs (TC) and security or liquidity constraints.
On the other hand, problem definition constraints take into account
conditions related to the formulation of the optimization problem, such
cardinality constraints (CC) and boundary constraints (BC). A more
detailed review of PO constraints can be found in
[4; 5; @milhomem2020analysis].

**Table 1: Main real-world constraints used in Portfolio Selection**

| **Hard Constraints**         | **Construction Constraints**      | **Problem Constraints**              |
|------------------------------|-----------------------------------|--------------------------------------|
| Budget Constraint             | Negative Weights (Long-Short)    | Cardinality Constraints (CC)         |
| Non-negativity                | Bankruptcy Probability            | Boundary Constraints (BC)            |
|                              | Transaction Costs (TC)           | Transaction Lots                     |
|                              | Tradeability Constraints          | Roundlot (Minimum Lots)             |
|                              | Liquidity Constraints             | Turnover Constraints                 |
|                              | Security Constraints              |                                      |
|                              | Market Scenario Constraints       |                                      |
|                              | Investment-Horizon Constraints     |                                      |
                                                   

<br/>

## 6) Portfolio Optimization Solution Techniques

The original MVPO problem possess a quadratic structure, which makes it
addressable by exact solution algorithms. Exact solution techniques are
attractive because of their ease of implementation and ability to find
the global optima of the PO problem [@milhomem2020analysis]. The most
popular exact technique to address the PSP in its classical MV form has
been quadratic programming (QP), which is a particular type of nonlinear
programming (nLP). However, when additional integer constraints are
considered (such as TC, CC or BC) the MVPO may transform into a
non-convex problem where QP is no longer applicable
[4]. In this case, alternative nLP exact
techniques exist to address the problem, but they can pose challenges in
analytical and computational terms.

In this context, rather than using nLP an increasing number of authors
prefer addressing the problem using approximate techniques such as
machine learning (ML), metaheuristics (MH) or hybrid models. Although
they do not guarantee to find the optimal solution, approximate
techniques often lead to a close-to-optimal solution with lower
computational expense than exact methods [@moral2006selection]. Further
details regarding approximate techniques are presented in the next
section.

</br>

# References
1. Aburto, L., Romero-Romero, R., Linfati, R., Escobar, J.W., et al.: An approach for a multi-period portfolio selection problem by considering transaction costs and prediction on the stock market. Complexity (2023)
2. Deb, K.: Multi-Objective Optimization, pp. 273–316. Springer US (2005)
3. Horn, J., Nafpliotis, N., Goldberg, D.E.: A niched pareto genetic algorithm for
multiobjective optimization. In: IEEE, pp. 82–87 (1994)
4. Kalayci, C.B., Ertenlice, O., Akbay, M.A.: A comprehensive review of deterministic models and applications for mean-variance portfolio optimization. Expert Systems
with Applications 125 pp. 345–368 (2019)
5. Kolm,P.N.,Tutuncu,R.,Fabozzi,F.J.: 60 years of portfolio optimization: Practical
challenges and current trends. European Journal of Operational Research (2014)
6. Konno, H., Shirakawa, H., Yamazaki, H.: A mean-absolute deviation-skewness portfolio optimization model. Annals of Operations Research 45 pp. 205–220 (1993)
7. Ma, Y., Han, R., Wang, W.: Portfolio optimization with return prediction using deep learning and machine learning. Expert Systems with Applications 165 p.
113973 (2021)
8. Markowitz, H.M., et al.: Portfolio selection. The Journal of Finance (1952)
9. Roman, D., Mitra, G.: Portfolio selection models: a review and new directions. The
international journal of innovative quantitative finance research pp. 69–85 (2009)
10. Sánchez-Fernández, Álvaro, Javier Díez-González, and Hilde Perez. "Artificial Intelligence in Portfolio Selection Problem: A Review and Future Perspectives." International Conference on Hybrid Artificial Intelligence Systems. Cham: Springer Nature Switzerland, 2024
11. Zhang, Y., Li, X., Guo, S.: Portfolio selection problems with markowitz’s mean–
variance framework: a review of literature. Fuzzy Optimization and Decision Mak-
ing 17 pp. 125–158 (2018)
12. Zitzler, E., Thiele, L.: Multiobjective evolutionary algorithms: a comparative case
study and the strength pareto approach. IEEE transactions on Evolutionary Com- putation 1 (4), 257–271 (1999)




