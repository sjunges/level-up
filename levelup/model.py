import logging
import time

import stormpy as sp
import stormpy.pars as spp

logger = logging.getLogger(__name__)


class HierarchicalModelDescription:
    def __init__(self, system_model, step_model, system_variables_to_step_constants, reward_name, system_formula, step_formula, system_formula_lb ):
        start_time = time.monotonic()
        self.system_model = system_model
        self._binary_output = not system_model.labeling.contains_label("stepstates")

        if self._binary_output:
            self.win_variable = [var for var in system_model.collect_all_parameters() if var.name == "win"][0]

            self.system_model_step_states = system_model.get_states_with_parameter(self.win_variable)
        else:
            self.win_variable = None
            self.system_model_step_states = system_model.labeling.get_states("stepstates")

        self.step_parametric_model = step_model
        self.instantiation_builder = spp.PMdpInstantiator(self.step_parametric_model)
        step_parameters_list = self.step_parametric_model.collect_all_parameters()
        step_parameters = {p.name: p for p in step_parameters_list}

        self.system_variables_to_step_parameters = {var: step_parameters[const.name] for var, const in
                                               system_variables_to_step_constants.items()}
        self.reward_name = reward_name
        self.step_formula = step_formula
        self.system_formula_lb = system_formula_lb
        self.system_formula = system_formula
        state_vals = self.system_model.state_valuations
        self.valuation_to_id = dict()
        self.state_id_to_valuation_id = dict()
        self.valuation_id_to_state_ids = dict()
        self.valuations = []

        for state_id in self.system_model_step_states:
            valuation = dict()
            for var, par in self.system_variables_to_step_parameters.items():
                valuation[par] = sp.RationalRF(state_vals.get_integer_value(state_id, var))
            tvaluation = tuple(valuation.values())
            val_id = self.valuation_to_id.setdefault(tvaluation, len(self.valuations))
            if val_id == len(self.valuations):
                self.valuations.append(valuation)
                self.valuation_id_to_state_ids[val_id] = [state_id]
            else:
                self.valuation_id_to_state_ids[val_id].append(state_id)
            self.state_id_to_valuation_id[state_id] = val_id
        self._build_time = time.monotonic() - start_time

    @property
    def has_single_outputs(self):
        return not self._binary_output

    @property
    def model_stats(self):
        return {"subMDP-states" : self.step_parametric_model.nr_states,
                "subMDP-choices": self.step_parametric_model.nr_choices,
                "nr-valuations": len(self.valuation_to_id),
                "macroMDP-states": self.system_model.nr_states,
                "macroMDP-choices": self.system_model.nr_choices,
                "nr-template-parameters": len(self.step_parametric_model.collect_all_parameters()),
                "startup-time": self._build_time}

    def instantiate_step_model(self, valuation):
        return self.instantiation_builder.instantiate(valuation)
