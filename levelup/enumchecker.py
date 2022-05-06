import logging
import time

import stormpy
import stormpy as sp

logger = logging.getLogger(__name__)


def _evaluate_aux(ratfunc, oldvar, res):
    assert ratfunc.denominator.is_constant
    winval = sp.RationalRF(res)
    numval = ratfunc.numerator.polynomial().evaluate({oldvar: winval})
    denom = ratfunc.denominator.constant_part()
    return sp.Rational(numval / denom)


def _to_exact_reward_model(parametric_reward_model):
    assert not parametric_reward_model.has_state_action_rewards
    assert not parametric_reward_model.has_transition_rewards
    state_reward = [sp.Rational(x.constant_part()) for x in parametric_reward_model.state_rewards]
    return stormpy.SparseExactRewardModel(optional_state_reward_vector=state_reward)


class FullInstantiationCheckerStats:
    def __init__(self):
        self._system_mc_timing = None
        self._instantiation_timing = None
        self._different_valuations = None

    def set_system_mc_time(self, t):
        self._system_mc_timing = t

    def set_instantiation_time(self, t):
        self._instantiation_timing = t

    def set_different_valuations(self, n):
        self._different_valuations = n

    def to_dict(self):
        return {"system mc time": self._system_mc_timing,
                "instantiations time": self._instantiation_timing,
                "different valuations": self._different_valuations}

    def __str__(self):
        return str(self.to_dict())


class SingleOutputFullInstantiationChecker:
    def __init__(self, hmd):
        self.hmd = hmd
        self._stats = FullInstantiationCheckerStats()

    @property
    def stats(self):
        return self._stats

    def run(self):
        start_time = time.monotonic()
        cache = dict()
        for state_id in self.hmd.system_model_step_states:
            val_id = self.hmd.state_id_to_valuation_id[state_id]
            cached_result = cache.get(val_id)
            if cached_result is None:
                valuation = self.hmd.valuations[val_id]
                cached_result = sp.model_checking(self.hmd.instantiation_builder.instantiate(valuation), self.hmd.step_formula).at(
                    self.hmd.step_parametric_model.initial_states[0])
                logger.debug(f"For valuation {valuation}, obtain {cached_result}.")
                cache[val_id] = cached_result
            if cached_result is not None:
                self.hmd.system_model.get_reward_model(self.hmd.reward_name).set_state_reward(state_id, cached_result)
        logger.debug(f"Model-step states: {self.hmd.system_model_step_states.number_of_set_bits()}")
        logger.debug(f"Different valuations: {len(self.hmd.valuations)}")
        self._stats.set_different_valuations(len(self.hmd.valuations))
        instantiation_time = time.monotonic() - start_time
        self._stats.set_instantiation_time(instantiation_time)
        result = sp.model_checking(self.hmd.system_model, self.hmd.system_formula).at(self.hmd.system_model.initial_states[0])
        system_mc_time = time.monotonic() - start_time - instantiation_time
        self._stats.set_system_mc_time(system_mc_time)
        logger.debug(f"Result: {result} in {system_mc_time+instantiation_time}s")
        logger.debug("--------------")
        return result


class BinaryOutputFullInstantiationChecker:
    def __init__(self, hmd):
        self.hmd = hmd
        self._stats = FullInstantiationCheckerStats()

    @property
    def stats(self):
        return self._stats

    def run(self):
        start_time = time.monotonic()
        cache = dict()

        new_transition_matrix_builder = sp.ExactSparseMatrixBuilder(self.hmd.system_model.nr_choices,
                                                                         self.hmd.system_model.nr_states,
                                                                         self.hmd.system_model.transition_matrix.nr_entries,
                                                                         has_custom_row_grouping=True)
        for state_id in range(self.hmd.system_model.nr_states):
            new_transition_matrix_builder.new_row_group(state_id)
            first_row = self.hmd.system_model.transition_matrix.get_row_group_start(state_id)
            beyond_last_row = self.hmd.system_model.transition_matrix.get_row_group_end(state_id)
            for row in range(first_row, beyond_last_row):
                for entry in self.hmd.system_model.transition_matrix.get_row(row):
                    if self.hmd.system_model_step_states.get(state_id):
                        val_id = self.hmd.state_id_to_valuation_id[state_id]
                        cached_result = cache.get(val_id)
                        if cached_result is None:
                            valuation = self.hmd.valuations[val_id]
                            cached_result = sp.model_checking(self.hmd.instantiation_builder.instantiate(valuation),
                                                              self.hmd.step_formula).at(
                                self.hmd.step_parametric_model.initial_states[0])
                            logger.debug(f"For valuation {valuation}, obtain {cached_result}.")
                            cache[val_id] = cached_result
                        new_transition_matrix_builder.add_next_value(row, entry.column,
                                                                     _evaluate_aux(entry.value(), self.hmd.win_variable,
                                                                                     cache[val_id]))
                    else:
                        new_transition_matrix_builder.add_next_value(row, entry.column, sp.Rational(entry.value().constant_part()))

        reward_models = {}
        for name, rm in self.hmd.system_model.reward_models.items():
            reward_models[name] = _to_exact_reward_model(rm)

        model_components = sp.SparseExactModelComponents(
            transition_matrix=new_transition_matrix_builder.build(),
            state_labeling=self.hmd.system_model.labeling,
            reward_models=reward_models,
            rate_transitions=False)

        instantiated_model = sp.SparseExactMdp(model_components)

        logger.debug(f"Model-step states: {self.hmd.system_model_step_states.number_of_set_bits()}")
        logger.debug(f"Different valuations: {len(self.hmd.valuations)}")
        instantiation_time = time.monotonic() - start_time
        self._stats.set_instantiation_time(instantiation_time)
        result = sp.model_checking(instantiated_model, self.hmd.system_formula).at(instantiated_model.initial_states[0])
        system_mc_time = time.monotonic() - start_time - instantiation_time
        self._stats.set_system_mc_time(system_mc_time)
        logger.debug(f"Result: {result} in {system_mc_time+instantiation_time}s")
        logger.debug("--------------")
        return result