"""
``BaseConvexOptimizer`` is the base class for all ``cvxpy`` (and ``scipy``)
optimization.
"""

import copy
import warnings

import numpy as np
import scipy.optimize as sco
import cvxpy as cp

from .base_optimizer import BaseOptimizer
from pypo.utils import exceptions
from pypo.utils.optimization_utils import _get_all_args

class BaseConvexOptimizer(BaseOptimizer):

    """
    The BaseConvexOptimizer contains many private variables for use by
    ``cvxpy``. For example, the immutable optimization variable for weights
    is stored as self._w. Interacting directly with these variables directly
    is discouraged.

    Instance variables:

    - ``n_assets`` - int
    - ``tickers`` - str list
    - ``weights`` - np.ndarray
    - ``_opt`` - cp.Problem
    - ``_solver`` - str
    - ``_solver_options`` - {str: str} dict

    Public methods:

    - ``add_objective()`` adds a (convex) objective to the optimization problem
    - ``add_constraint()`` adds a constraint to the optimization problem
    - ``convex_objective()`` solves for a generic convex objective with linear constraints
    - ``nonconvex_objective()`` solves for a generic nonconvex objective using the scipy backend.
      This is prone to getting stuck in local minima and is generally *not* recommended.
    - ``set_weights()`` creates self.weights (np.ndarray) from a weights dict
    - ``clean_weights()`` rounds the weights and clips near-zeros.
    - ``save_weights_to_file()`` saves the weights to csv, json, or txt.
    """

    def __init__(
        self,
        n_assets,
        tickers=None,
        weight_bounds=(0, 1),
        solver=None,
        verbose=False,
        solver_options=None,
    ):
        """
        :param weight_bounds: minimum and maximum weight of each asset OR single min/max pair
                              if all identical, defaults to (0, 1). Must be changed to (-1, 1)
                              for portfolios with shorting.
        :type weight_bounds: tuple OR tuple list, optional
        :param solver: name of solver. list available solvers with: ``cvxpy.installed_solvers()``
        :type solver: str, optional.
        :param verbose: whether performance and debugging info should be printed, defaults to False
        :type verbose: bool, optional
        :param solver_options: parameters for the given solver
        :type solver_options: dict, optional
        """
        super().__init__(n_assets, tickers)

        # Optimization variables
        self._w = cp.Variable(n_assets)
        self._objective = None
        self._additional_objectives = []
        self._constraints = []
        self._lower_bounds = None
        self._upper_bounds = None
        self._opt = None
        self._solver = solver
        self._verbose = verbose
        self._solver_options = solver_options if solver_options else {}
        self._map_bounds_to_constraints(weight_bounds)

    def deepcopy(self):
        """
        Returns a custom deep copy of the optimizer. This is necessary because
        ``cvxpy`` expressions do not support deepcopy, but the mutable arguments need to be
        copied to avoid unintended side effects. Instead, we create a shallow copy
        of the optimizer and then manually copy the mutable arguments.
        """
        self_copy = copy.copy(self)
        self_copy._additional_objectives = [
            copy.copy(obj) for obj in self_copy._additional_objectives
        ]
        self_copy._constraints = [copy.copy(con) for con in self_copy._constraints]
        return self_copy

    def _map_bounds_to_constraints(self, test_bounds):
        """
        Convert input bounds into a form acceptable by cvxpy and add to the constraints list.

        :param test_bounds: minimum and maximum weight of each asset OR single min/max pair
                            if all identical OR pair of arrays corresponding to lower/upper bounds. defaults to (0, 1).
        :type test_bounds: tuple OR list/tuple of tuples OR pair of np arrays
        :raises TypeError: if ``test_bounds`` is not of the right type
        :return: bounds suitable for cvxpy
        :rtype: tuple pair of np.ndarray
        """
        # If it is a collection with the right length, assume they are all bounds.
        if len(test_bounds) == self.n_assets and not isinstance(
            test_bounds[0], (float, int)
        ):
            bounds = np.array(test_bounds, dtype=float)
            self._lower_bounds = np.nan_to_num(bounds[:, 0], nan=-np.inf)
            self._upper_bounds = np.nan_to_num(bounds[:, 1], nan=np.inf)
        else:
            # Otherwise this must be a pair.
            if len(test_bounds) != 2 or not isinstance(test_bounds, (tuple, list)):
                raise TypeError(
                    "test_bounds must be a pair (lower bound, upper bound) OR a collection of bounds for each asset"
                )
            lower, upper = test_bounds

            # Replace None values with the appropriate +/- 1
            if np.isscalar(lower) or lower is None:
                lower = -1 if lower is None else lower
                self._lower_bounds = np.array([lower] * self.n_assets)
                upper = 1 if upper is None else upper
                self._upper_bounds = np.array([upper] * self.n_assets)
            else:
                self._lower_bounds = np.nan_to_num(lower, nan=-1)
                self._upper_bounds = np.nan_to_num(upper, nan=1)

        self.add_constraint(lambda w: w >= self._lower_bounds)
        self.add_constraint(lambda w: w <= self._upper_bounds)

    def is_parameter_defined(self, parameter_name: str) -> bool:
        is_defined = False
        objective_and_constraints = (
            self._constraints + [self._objective]
            if self._objective is not None
            else self._constraints
        )
        for expr in objective_and_constraints:
            params = [
                arg for arg in _get_all_args(expr) if isinstance(arg, cp.Parameter)
            ]
            for param in params:
                if param.name() == parameter_name and not is_defined:
                    is_defined = True
                elif param.name() == parameter_name and is_defined:
                    raise exceptions.InstantiationError(
                        "Parameter name defined multiple times"
                    )
        return is_defined

    def update_parameter_value(self, parameter_name: str, new_value: float) -> None:
        if not self.is_parameter_defined(parameter_name):
            raise exceptions.InstantiationError("Parameter has not been defined")
        was_updated = False
        objective_and_constraints = (
            self._constraints + [self._objective]
            if self._objective is not None
            else self._constraints
        )
        for expr in objective_and_constraints:
            params = [
                arg for arg in _get_all_args(expr) if isinstance(arg, cp.Parameter)
            ]
            for param in params:
                if param.name() == parameter_name:
                    param.value = new_value
                    was_updated = True
        if not was_updated:
            raise exceptions.InstantiationError("Parameter was not updated")

    def _solve_cvxpy_opt_problem(self):
        """
        Helper method to solve the cvxpy problem and check output,
        once objectives and constraints have been defined

        :raises exceptions.OptimizationError: if problem is not solvable by cvxpy
        """
        try:
            if self._opt is None:
                self._opt = cp.Problem(cp.Minimize(self._objective), self._constraints)
                self._initial_objective = self._objective.id
                self._initial_constraint_ids = {const.id for const in self._constraints}
            else:
                if not self._objective.id == self._initial_objective:
                    raise exceptions.InstantiationError(
                        "The objective function was changed after the initial optimization. "
                        "Please create a new instance instead."
                    )

                constr_ids = {const.id for const in self._constraints}
                if not constr_ids == self._initial_constraint_ids:
                    raise exceptions.InstantiationError(
                        "The constraints were changed after the initial optimization. "
                        "Please create a new instance instead."
                    )
            self._opt.solve(
                solver=self._solver, verbose=self._verbose, **self._solver_options
            )

        except (TypeError, cp.DCPError) as e:
            raise exceptions.OptimizationError from e

        if self._opt.status not in {"optimal", "optimal_inaccurate"}:
            raise exceptions.OptimizationError(
                "Solver status: {}".format(self._opt.status)
            )
        self.weights = self._w.value.round(16) + 0.0  # +0.0 removes signed zero
        return self._make_output_weights()

    def add_objective(self, new_objective, **kwargs):
        """
        Add a new term into the objective function. This term must be convex,
        and built from cvxpy atomic functions.

        Example::

            def L1_norm(w, k=1):
                return k * cp.norm(w, 1)

            ef.add_objective(L1_norm, k=2)

        :param new_objective: the objective to be added
        :type new_objective: cp.Expression (i.e function of cp.Variable)
        """
        if self._opt is not None:
            raise exceptions.InstantiationError(
                "Adding objectives to an already solved problem might have unintended consequences. "
                "A new instance should be created for the new set of objectives."
            )
        self._additional_objectives.append(new_objective(self._w, **kwargs))

    def add_constraint(self, new_constraint):
        """
        Add a new constraint to the optimization problem. This constraint must satisfy DCP rules,
        i.e be either a linear equality constraint or convex inequality constraint.

        Examples::

            ef.add_constraint(lambda x : x[0] == 0.02)
            ef.add_constraint(lambda x : x >= 0.01)
            ef.add_constraint(lambda x: x <= np.array([0.01, 0.08, ..., 0.5]))

        :param new_constraint: the constraint to be added
        :type new_constraint: callable (e.g lambda function)
        """
        if not callable(new_constraint):
            raise TypeError(
                "New constraint must be provided as a callable (e.g lambda function)"
            )
        if self._opt is not None:
            raise exceptions.InstantiationError(
                "Adding constraints to an already solved problem might have unintended consequences. "
                "A new instance should be created for the new set of constraints."
            )
        self._constraints.append(new_constraint(self._w))

    def add_sector_constraints(self, sector_mapper, sector_lower, sector_upper):
        """
        Adds constraints on the sum of weights of different groups of assets.
        Most commonly, these will be sector constraints e.g portfolio's exposure to
        tech must be less than x%::

            sector_mapper = {
                "GOOG": "tech",
                "FB": "tech",,
                "XOM": "Oil/Gas",
                "RRC": "Oil/Gas",
                "MA": "Financials",
                "JPM": "Financials",
            }

            sector_lower = {"tech": 0.1}  # at least 10% to tech
            sector_upper = {
                "tech": 0.4, # less than 40% tech
                "Oil/Gas": 0.1 # less than 10% oil and gas
            }

        :param sector_mapper: dict that maps tickers to sectors
        :type sector_mapper: {str: str} dict
        :param sector_lower: lower bounds for each sector
        :type sector_lower: {str: float} dict
        :param sector_upper: upper bounds for each sector
        :type sector_upper: {str:float} dict
        """
        if np.any(self._lower_bounds < 0):
            warnings.warn(
                "Sector constraints may not produce reasonable results if shorts are allowed."
            )
        for sector in sector_upper:
            is_sector = [sector_mapper.get(t) == sector for t in self.tickers]
            self.add_constraint(lambda w: cp.sum(w[is_sector]) <= sector_upper[sector])
        for sector in sector_lower:
            is_sector = [sector_mapper.get(t) == sector for t in self.tickers]
            self.add_constraint(lambda w: cp.sum(w[is_sector]) >= sector_lower[sector])

    def convex_objective(self, custom_objective, weights_sum_to_one=True, **kwargs):
        """
        Optimize a custom convex objective function. Constraints should be added with
        ``ef.add_constraint()``. Optimizer arguments must be passed as keyword-args. Example::

            # Could define as a lambda function instead
            def logarithmic_barrier(w, cov_matrix, k=0.1):
                # 60 Years of Portfolio Optimization, Kolm et al (2014)
                return cp.quad_form(w, cov_matrix) - k * cp.sum(cp.log(w))

            w = ef.convex_objective(logarithmic_barrier, cov_matrix=ef.cov_matrix)

        :param custom_objective: an objective function to be MINIMISED. This should be written using
                                 cvxpy atoms Should map (w, `**kwargs`) -> float.
        :type custom_objective: function with signature (cp.Variable, `**kwargs`) -> cp.Expression
        :param weights_sum_to_one: whether to add the default objective, defaults to True
        :type weights_sum_to_one: bool, optional
        :raises OptimizationError: if the objective is nonconvex or constraints nonlinear.
        :return: asset weights for the efficient risk portfolio
        :rtype: OrderedDict
        """
        # custom_objective must have the right signature (w, **kwargs)
        self._objective = custom_objective(self._w, **kwargs)

        for obj in self._additional_objectives:
            self._objective += obj

        if weights_sum_to_one:
            self.add_constraint(lambda w: cp.sum(w) == 1)

        return self._solve_cvxpy_opt_problem()

    def nonconvex_objective(
        self,
        custom_objective,
        objective_args=None,
        weights_sum_to_one=True,
        constraints=None,
        solver="SLSQP",
        initial_guess=None,
    ):
        """
        Optimize some objective function using the scipy backend. This can
        support nonconvex objectives and nonlinear constraints, but may get stuck
        at local minima. Example::

            # Market-neutral efficient risk
            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w)},  # weights sum to zero
                {
                    "type": "eq",
                    "fun": lambda w: target_risk ** 2 - np.dot(w.T, np.dot(ef.cov_matrix, w)),
                },  # risk = target_risk
            ]
            ef.nonconvex_objective(
                lambda w, mu: -w.T.dot(mu),  # min negative return (i.e maximise return)
                objective_args=(ef.expected_returns,),
                weights_sum_to_one=False,
                constraints=constraints,
            )

        :param objective_function: an objective function to be MINIMISED. This function
                                   should map (weight, args) -> cost
        :type objective_function: function with signature (np.ndarray, args) -> float
        :param objective_args: arguments for the objective function (excluding weight)
        :type objective_args: tuple of np.ndarrays
        :param weights_sum_to_one: whether to add the default objective, defaults to True
        :type weights_sum_to_one: bool, optional
        :param constraints: list of constraints in the scipy format (i.e dicts)
        :type constraints: dict list
        :param solver: which SCIPY solver to use, e.g "SLSQP", "COBYLA", "BFGS".
                       User beware: different optimizers require different inputs.
        :type solver: string
        :param initial_guess: the initial guess for the weights, shape (n,) or (n, 1)
        :type initial_guess: np.ndarray
        :return: asset weights that optimize the custom objective
        :rtype: OrderedDict
        """
        # Sanitise inputs
        if not isinstance(objective_args, tuple):
            objective_args = (objective_args,)

        # Make scipy bounds
        bound_array = np.vstack((self._lower_bounds, self._upper_bounds)).T
        bounds = list(map(tuple, bound_array))

        if initial_guess is None:
            initial_guess = np.array([1 / self.n_assets] * self.n_assets)

        # Construct constraints
        final_constraints = []
        if weights_sum_to_one:
            final_constraints.append({"type": "eq", "fun": lambda w: np.sum(w) - 1})
        if constraints is not None:
            final_constraints += constraints

        result = sco.minimize(
            custom_objective,
            x0=initial_guess,
            args=objective_args,
            method=solver,
            bounds=bounds,
            constraints=final_constraints,
        )
        self.weights = result["x"]
        return self._make_output_weights()
