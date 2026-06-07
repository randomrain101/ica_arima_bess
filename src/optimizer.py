from dataclasses import dataclass

import pyomo.environ as pyo

import numpy as np


@dataclass
class OptimizerBESS:
    # Number of consecutive products / time blocks to optimize over
    n_products: int = 24
    # Initial SOC of the battery (MWh)
    soc_init: float = None
    # Energy (Capacity) in MWh
    energy: float = 2
    # Power in MW
    power: float = 1
    # Maximum cylcles allowed for n_products period
    max_cycles: float = 1.5

    @staticmethod
    def soc_min_rule(model, t):
        # SOC can never be negative (positive schedule discharges and reduces SOC, negative schedule charges and increases SOC)
        return model.soc_init + sum(-model.schedule[i] for i in range(t + 1)) >= 0

    @staticmethod
    def soc_max_rule(model, t):
        # SOC can never be bigger than energy limit (capacity)
        return model.soc_init + sum(-model.schedule[i] for i in range(t + 1)) <= model.energy
    
    @staticmethod
    def same_start_end_soc(model):
        return sum(model.schedule[t] for t in model.T) == 0

    @staticmethod
    def pos_rule(model, t):
        # pos_val[t] >= schedule[t], combined with non-negativity gives pos_val[t] = max(0, schedule[t])
        return model.pos_val[t] >= model.schedule[t]

    @staticmethod
    def max_cycles_rule(model):
        # One cycle = full discharge (energy MWh); sum of positive schedules equals discharged energy
        # Valid because same_start_end_soc ensures sum(positive) == sum(|negative|)
        return sum(model.pos_val[t] for t in model.T) / model.energy <= model.max_cycles

    # Require array of prices with shape (n_products)
    def optimize(self, prices: np.array):
        assert prices.shape[0] == self.n_products and len(prices.shape) == 1
        # Schedule can be positive (Discharge -> Sell) or negative (Charge -> Buy), in the range (-power, power)
        model = pyo.ConcreteModel()
        model.T = pyo.RangeSet(0, self.n_products - 1)
        model.schedule = pyo.Var(model.T, bounds=(-self.power, self.power), initialize=0, domain=pyo.Reals)
        # Auxiliary variables for the positive part of schedule[t], i.e. max(0, schedule[t])
        model.pos_val = pyo.Var(model.T, bounds=(0, self.power), initialize=0, domain=pyo.NonNegativeReals)
        model.energy = self.energy
        model.soc_init = self.soc_init
        model.max_cycles = self.max_cycles
        # Optimize profit
        model.profit = pyo.Objective(
            expr=sum(prices[i] * model.schedule[i] for i in model.T),
            sense=pyo.maximize,
        )
        model.min_soc_rule = pyo.Constraint(model.T, rule=self.soc_min_rule)
        model.max_soc_rule = pyo.Constraint(model.T, rule=self.soc_max_rule)
        model.pos_con = pyo.Constraint(model.T, rule=self.pos_rule)
        model.max_cycles_rule = pyo.Constraint(rule=self.max_cycles_rule)
        model.same_start_end_soc_con = pyo.Constraint(rule=self.same_start_end_soc)

        results = pyo.SolverFactory("glpk").solve(model, tee=False)
        if results.solver.status == 'ok':
            return np.array([pyo.value(model.schedule[i]) for i in model.T])
        
        