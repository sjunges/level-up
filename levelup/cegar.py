import copy
import logging
import math
import heapq
import os
import time
import queue

import matplotlib.pyplot as plt

import pycarl as pc
import stormpy
import stormpy as sp
import stormpy.pars as spp

logger = logging.getLogger(__name__)


def region_contains_valuation(region_bounds, valuation):
    for dim, val in zip(region_bounds.values(), valuation):
        if val < dim[0] or val > dim[1]:
            return False
    return True


class AnnotatedRegion:
    counter = 0
    valuations = []

    def __init__(self, region_bounds, valuation_indices, induced_lb, induced_ub, weight):
        self.region_bounds = region_bounds
        self.pts = set(valuation_indices)
        self._induced_lb = induced_lb
        self._induced_ub = induced_ub
        self.counter += 1
        self._age = self.counter
        self._weight = weight
        self._eliminate_pts = set()

    def __lt__(self, other):
        if self.priority == other.priority:
            return self._age < other._age
        else:
            return self.priority > other.priority

    @property
    def parameter_region(self):
        return spp.ParameterRegion(self.region_bounds)

    def set_induced_bounds(self, lb, ub):
        assert lb <= ub
        self._induced_lb = lb
        self._induced_ub = ub

    def get_induced_bounds(self):
        return self._induced_lb, self._induced_ub

    def contains_valuation(self, valuation):
        for dim, val in zip(self.region_bounds.values(), valuation.values()):
            if val < dim[0] or val > dim[1]:
                return False
        return True

    def contains_valuation_id(self, id):
        return id in self.pts

    @property
    def priority(self):
        gap = self._induced_ub - self._induced_lb
        return gap * (1 + sp.RationalRF(float(len(self.pts))/len(self.valuations)) * sp.RationalRF(1) + sp.RationalRF(float(self._weight)/len(self.valuations)) * sp.RationalRF(2))

    def shrink(self):
        var_lb = {par: sp.RationalRF(1000000000) for par in self.region_bounds.keys()}
        var_ub = {par: -sp.RationalRF(1000000000) for par in self.region_bounds.keys()}
        for x in self.pts:
            if x in self._eliminate_pts:
                continue
            for par in self.region_bounds.keys():
                var_lb[par] = min(var_lb[par], self.valuations[x][par])
                var_ub[par] = max(var_ub[par], self.valuations[x][par])
                assert var_lb[par] <= var_ub[par]
        self.region_bounds = {par: [var_lb[par], var_ub[par]] for par in self.region_bounds.keys()}

    def eliminate_pts(self, indices):
        for i in indices:
            self._eliminate_pts.add(i)

    def split(self, point_weights):
        logger.debug(f"Splitting: {str(self)}")
        if len(self.pts) <= 1:
            return []
        maxdim = None
        maxrange = 0
        for dim, range in self.region_bounds.items():
            if range[1] - range[0] > maxrange:
                maxdim = dim
                maxrange = range[1] - range[0]
        if maxrange == 0:
            return []
        dim = maxdim
        assert dim in self.region_bounds
        new_region_bounds_lhs = {par: copy.copy(bounds) for par, bounds in self.region_bounds.items()}
        new_region_bounds_rhs = {par: copy.copy(bounds) for par, bounds in self.region_bounds.items()}
        splitpoint = (self.region_bounds[dim][0] + self.region_bounds[dim][1]) * sp.RationalRF(0.5)
        logger.debug(f"Splitting around {splitpoint} in dimension {dim}")
        new_region_bounds_lhs[dim][1] = splitpoint
        new_region_bounds_rhs[dim][0] = splitpoint
        pts_lhs = []
        pts_rhs = []
        for pt in self.pts:
            if pt in self._eliminate_pts:
                continue
            if self.valuations[pt][dim] < splitpoint:
                pts_lhs.append(pt)
                lhs_weight = point_weights[pt]
            else:
                pts_rhs.append(pt)
                rhs_weight = point_weights[pt]
        assert len(pts_lhs) + len(pts_rhs) <= len(self.pts)#, f"{len(pts_lhs)} + {len(pts_rhs)} + {len(self._eliminate_pts)} == {len(self.pts)}"
        return [AnnotatedRegion(new_region_bounds_lhs, pts_lhs, self._induced_lb, self._induced_ub, lhs_weight),
                AnnotatedRegion(new_region_bounds_rhs, pts_rhs, self._induced_lb, self._induced_ub, rhs_weight)]

    def __str__(self):
        return f"{self.parameter_region}:<prio={float(self.priority):.2f}, gap={float(self._induced_ub -  self._induced_lb)}, pts={len(self.pts)}>"


class CegarCheckerOptions:
    def __init__(self, max_reach_steps, reassesement_iterations, compute_expected_visits, acceptable_gap):
        self.max_reach_steps = max_reach_steps
        self.reassesement_iterations = reassesement_iterations
        self.compute_expected_number_of_visits = compute_expected_visits
        self.nlargest_weights_individually_percent = 1
        self.acceptable_gap = acceptable_gap


class CegarCheckerStats:
    def __init__(self, do_cache_valuations=False):
        self._global_lower_bound_results = []
        self._global_upper_bound_results = []
        self._timing = []
        self._nr_cached_valuations = []
        self._cached_valuations = []
        self._do_cache_valuations = do_cache_valuations
        self._start_time = None
        self._start_analyse_submodel_time = None
        self._start_parametric_step_mc_time = None
        self._parametric_step_mc_time = 0
        self._analyse_submodel_timing = 0
        self._submodel_lower_bound = None
        self._submodel_upper_bound = None
        self._submodel_set_analyses = 0
        self._start_analyse_system_time = None
        self._analyse_system_time = 0
        self._number_of_system_analyses = 0
        self._number_of_sampled_step_models = 0
        self._start_sample_step_time = None
        self._sample_step_time = 0
        self._p50time = None
        self._p90time = None

    def add_new_global_result(self, lower_bound, upper_bound, cached_valuations):
        complete_time = time.monotonic() - self._start_time
        logger.debug(f"Obtained {lower_bound}--{upper_bound} in {complete_time}s")
        self._global_lower_bound_results.append(lower_bound)
        self._global_upper_bound_results.append(upper_bound)
        self._timing.append(complete_time)
        self._nr_cached_valuations.append(len(cached_valuations))
        if self._do_cache_valuations:
            self._cached_valuations.append(copy.deepcopy(cached_valuations))
        if self._p50time is None:
            if 0.5 * upper_bound <= lower_bound:
                self._p50time = time.monotonic() - self._start_time
        if self._p90time is None:
            if 0.9 * upper_bound <= lower_bound:
                self._p90time = time.monotonic() - self._start_time

    def start_sample_step_model(self):
        assert self._start_sample_step_time is None
        self._start_sample_step_time = time.monotonic()
        self._number_of_sampled_step_models += 1

    def end_sample_step_model(self):
        assert self._start_sample_step_time is not  None
        self._sample_step_time += time.monotonic() - self._start_sample_step_time
        self._start_sample_step_time = None

    def start_analyse_system(self):
        assert self._start_analyse_system_time is None
        self._start_analyse_system_time = time.monotonic()

    def end_analyse_system(self):
        assert self._start_analyse_system_time is not None
        self._analyse_system_time += time.monotonic() - self._start_analyse_system_time
        self._start_analyse_system_time = None
        self._number_of_system_analyses += 1

    def start_parametric_system_modelcheck(self):
        assert self._start_parametric_step_mc_time is None
        self._start_parametric_step_mc_time = time.monotonic()

    def end_parametric_system_modelcheck(self):
        assert self._start_parametric_step_mc_time is not None
        self._parametric_step_mc_time += time.monotonic() - self._start_parametric_step_mc_time
        self._start_parametric_step_mc_time = None

    def start_analyse_submodel(self, lower_bound, upper_bound):
        assert self._start_analyse_submodel_time is None
        self._start_analyse_submodel_time = time.monotonic()
        self._submodel_lower_bound = lower_bound
        self._submodel_upper_bound = upper_bound

    def end_analyse_submodel(self, lower_bound, upper_bound):
        assert self._start_analyse_submodel_time is not None
        checktime = time.monotonic() - self._start_analyse_submodel_time
        self._analyse_submodel_timing += checktime
        logger.debug(
            f"Improve [{float(self._submodel_lower_bound):.2f}, {float(self._submodel_upper_bound):.2f}] to [{float(lower_bound):.2f}, {float(upper_bound):.2f}] in {checktime}s")
        self._start_analyse_submodel_time = None
        self._submodel_lower_bound = None
        self._submodel_upper_bound = None
        self._submodel_set_analyses += 1

    def start(self):
        self._start_time = time.monotonic()

    def to_dict(self):
        return {"timing": self._timing[-1],
                "p50-timing": self._p50time,
                "p90-timing": self._p90time,
                "nr_system_analyses": self._number_of_system_analyses,
                "system_analysis_time": self._analyse_system_time,
                "system_parametric_mc_time": self._parametric_step_mc_time,
                "subsystem_set_analysis_time": self._analyse_submodel_timing,
                "nr_subsystem_analyses": self._submodel_set_analyses,
                "sample_subsystem_time": self._sample_step_time,
                "nr_subsystem_samples": self._number_of_sampled_step_models}

    def plot_timing(self, model_description="model", model_identifier="model", ground_truth=None, baseline_performance=None, output_directory="."):
        fig, ax = plt.subplots()
        ax.plot(self._timing, self._global_lower_bound_results, c="b")
        ax.plot(self._timing, self._global_upper_bound_results, c="b")
        if ground_truth is not None:
            ax.plot([0] + self._timing, [ground_truth]*(len(self._timing)+1), linestyle="dashed", c="k", linewidth=0.8)
        if baseline_performance is not None:
            ax.plot([baseline_performance, baseline_performance], [0, self._global_upper_bound_results[0]], c="m", linewidth=0.6, linestyle="dotted")
        ax.set_title(f"Quality of bounds over time for {model_description}")
        ax.set_xlabel("Time (s)")
        ax.set_xlim(0)
        ax.set_ylim(0)
        ax.set_ylabel("Exp. Cum. Reward")
        logger.info(f"Plot {model_identifier}-performance.png")
        plt.savefig(os.path.join(output_directory, f'{model_identifier}-performance.png'))

    def __str__(self):
        return str(self.to_dict())


class PrecomputedBoundUpdate:
    def __init__(self, valuations, lower_bound, upper_bound):
        self.valuations = valuations
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def get_lower_bound(self, val_id):
        return self.lower_bound

    def get_upper_bound(self, val_id):
        return self.upper_bound


class PrecomputedBoundsUpdate:
    def __init__(self, updates):
        self._updates = updates

    def get_lower_bound(self, val_id):
        return self._updates[val_id][0]

    def get_upper_bound(self, val_id):
        return self._updates[val_id][1]

    @property
    def valuations(self):
        return self._updates.keys()


class OnDemandValueUpdate:
    def __init__(self, valuations, computation):
        logger.debug(f"Create OnDemandValueUpdate with {len(valuations)} valuations")
        self.valuations = valuations
        self.computation = computation
        self._last_value = None
        self._last_val_id = None

    def get_lower_bound(self, val_id):
        return self._compute_value(val_id)

    def get_upper_bound(self, val_id):
        return self._compute_value(val_id)

    def _compute_value(self, val_id):
        if self._last_val_id != val_id:
            self._last_value = self.computation(val_id)
            self._last_val_id = val_id
        return self._last_value


def upto(i, j):
    if i > j:
        return j
    return i


class OnDemandBoundsUpdate:
    def __init__(self, valuations, computation):
        logger.debug(f"Create OnDemandBoundsUpdate with {len(valuations)} valuations")
        self.valuations = valuations
        self.computations = computation
        self._last_lower_bound_value = None
        self._last_upper_bound_value = None
        self._last_val_id = None

    def get_lower_bound(self, val_id):
        self._compute_bounds(val_id)
        return self._last_lower_bound_value

    def get_upper_bound(self, val_id):
        self._compute_bounds(val_id)
        return self._last_upper_bound_value

    def _compute_value(self, val_id):
        if self._last_val_id != val_id:
            self._last_lower_bound_value, self._last_upper_bound_value = self.computation(val_id)
            self._last_val_id = val_id


class CegarChecker:
    def __init__(self, hmd, options):
        self.hmd = hmd
        self._options = options
        self._stats = CegarCheckerStats()
        self._cache = dict()
        self._initialize()
        self._global_lb = 0
        self._global_ub = math.inf
        self._previous_lb_scheduler = None
        self._previous_ub_scheduler = None

    def _initialize(self):
        pass

    @property
    def stats(self):
        return self._stats

    def _analyse_system_model(self, update, extract_scheduler, only_lower_bound=False):
        raise RuntimeError("Abstract class")

    def _sample_step_model(self, val_id):
        cache_val = self._cache.get(val_id)
        if cache_val is not None:
            return cache_val
        self.stats.start_sample_step_model()
        res = sp.model_checking(self.hmd.instantiate_step_model(self.hmd.valuations[val_id]),
                          self.hmd.step_formula).at(
            self.hmd.step_parametric_model.initial_states[0])
        self._cache[val_id] = res
        self.stats.end_sample_step_model()
        return res

    def _compute_valuation_weigths(self, state_weights):
        weights = dict()
        for state_id in self.hmd.system_model_step_states:
            val_id = self.hmd.state_id_to_valuation_id.get(state_id)
            weights[val_id] = weights.get(val_id, 0) + state_weights.at(state_id)
        return weights

    def run(self):
        self._stats.start()

        state_vals = self.hmd.system_model.state_valuations

        # Check all instances
        var_lb = {var: sp.RationalRF(1000000000) for var in self.hmd.system_variables_to_step_parameters}
        var_ub = {var: -sp.RationalRF(1000000000) for var in self.hmd.system_variables_to_step_parameters}

        for state_id in self.hmd.system_model_step_states:
            for var, par in self.hmd.system_variables_to_step_parameters.items():
                val = sp.RationalRF(state_vals.get_integer_value(state_id, var))
                var_lb[var] = min(var_lb[var], val)
                var_ub[var] = max(var_ub[var], val)
                assert var_lb[var] <= var_ub[var]

        rcenv = sp.Environment()
        region_checker = spp.create_region_checker(rcenv, self.hmd.step_parametric_model, self.hmd.step_formula, False, True, preconditions_validated_manually=True)
        region_bounds = {par: [var_lb[var], var_ub[var]] for var, par in self.hmd.system_variables_to_step_parameters.items()}
        region = spp.ParameterRegion(region_bounds)
        self._stats.start_analyse_submodel(0, math.inf)
        upper_bound = region_checker.get_bound(rcenv, region, maximise=True).constant_part()
        lower_bound = region_checker.get_bound(rcenv, region, maximise=False).constant_part()
        self._stats.end_analyse_submodel(lower_bound, upper_bound)
        bound_update = PrecomputedBoundUpdate(self.hmd.valuation_id_to_state_ids.keys(), lower_bound, upper_bound)
        scheduler = self._analyse_system_model(bound_update, extract_scheduler=True)
        if scheduler:
            if self._options.compute_expected_number_of_visits:
                assert not scheduler.partial  # TODO: otherwise, model will not be a DTMC
                dtmc_system_model = self.hmd.system_model.apply_scheduler(scheduler,
                                                                          drop_unreachable_states=False)
                assert dtmc_system_model.nr_states == self.hmd.system_model.nr_states
                expected_number_of_visits = sp.compute_expected_number_of_visits(stormpy.Environment(),
                                                                                 dtmc_system_model)
                valuation_weights = self._compute_valuation_weigths(expected_number_of_visits)
                howmany = upto(int(self._options.nlargest_weights_individually_percent/100 * len(self.hmd.valuation_id_to_state_ids)),150)
                important_valuations = heapq.nlargest(howmany,
                                                      range(len(valuation_weights)), valuation_weights.get)

            else:
                important_valuations = []

            system_model_target_states = self.hmd.system_model.labeling.get_states("done")
            system_model_constraint_states = sp.BitVector(self.hmd.system_model.nr_states, True)

            reached_states = sp.get_reachable_states(self.hmd.system_model, self.hmd.system_model.initial_states_as_bitvector,
                                                     system_model_constraint_states, system_model_target_states,
                                                     maximal_steps=self._options.max_reach_steps,
                                                     choice_filter=scheduler.compute_action_support(
                                                         self.hmd.system_model.nondeterministic_choice_indices))
            reached_valuations = [self.hmd.state_id_to_valuation_id.get(state) for state in reached_states]

            individual_analysis = reached_valuations + important_valuations
        else:
            individual_analysis = []
        model_checking_call = lambda val_id: self._sample_step_model(val_id)
        bound_update = OnDemandValueUpdate(individual_analysis, model_checking_call)
        self._analyse_system_model(bound_update, extract_scheduler=False)
        logger.debug("Initialize loop...")
        Q = queue.PriorityQueue()
        AnnotatedRegion.valuations = self.hmd.valuations
        relevant_valuation_ids = set(self.hmd.valuation_id_to_state_ids.keys()) - set(self._cache.keys())

        for ar in AnnotatedRegion(region_bounds, relevant_valuation_ids, lower_bound, upper_bound, 0).split(valuation_weights):
            logger.debug(f"Adding {ar} to the queue for potential refinement.")
            Q.put(ar)
        i = 0 # outer loops
        j = 0 # inner loops with macro MDP checks
        results = []

        while not Q.empty():
            if self._global_lb * (1 + self._options.acceptable_gap) >= self._global_ub:
                logger.info(f"Done with [{self._global_lb}, {self._global_ub}] in {i} iterations.")
                break
            logger.info(f"Starting iteration {i + 1}...")

            current_annotated_region = Q.get()
            logger.debug(f"Current region: {current_annotated_region}")
            current_annotated_region.eliminate_pts(self._cache.keys())
            current_annotated_region.shrink()
            logger.debug(f"After shrinking region: {current_annotated_region.parameter_region}")
            preg = current_annotated_region.parameter_region

            old_lower_bound, old_upper_bound = current_annotated_region.get_induced_bounds()
            self._stats.start_analyse_submodel(old_lower_bound, old_upper_bound)
            upper_bound = region_checker.get_bound(rcenv, preg, maximise=True).constant_part()
            lower_bound = region_checker.get_bound(rcenv, preg, maximise=False).constant_part()
            self._stats.end_analyse_submodel(lower_bound, upper_bound)

            results.append(tuple([current_annotated_region, lower_bound, upper_bound]))
            current_annotated_region.set_induced_bounds(lower_bound, upper_bound)
            for reg in current_annotated_region.split(valuation_weights):
                logger.debug(f"Adding {reg}...")
                Q.put(reg)

            i = i + 1
            if i % self._options.reassesement_iterations == 0 or Q.empty():
                logger.debug("Consider system-level, shrinking global bounds...")
                j = j + 1
                region_to_reward_time = time.monotonic()
                _valuations_update = dict()
                for val_id in self.hmd.valuation_id_to_state_ids.keys():
                    if val_id in self._cache:
                        _valuations_update[val_id] = (self._cache[val_id], self._cache[val_id])
                    for entry in reversed(results):
                        reg, lb, ub = entry[0], entry[1], entry[2]
                        if reg.contains_valuation_id(val_id):
                            _valuations_update[val_id] = (lb, ub)
                            assert lb <= ub, f"Lower bound ({lb}) should not exceed upper bound ({ub})"
                            break
                region_to_reward_time = time.monotonic() - region_to_reward_time
                logger.debug(f"{region_to_reward_time}s")
                update = PrecomputedBoundsUpdate(_valuations_update)
                results = []

                scheduler = self._analyse_system_model(update, extract_scheduler=True, only_lower_bound=True)
                self._previous_lb_scheduler = scheduler
                if scheduler:
                    if self._options.compute_expected_number_of_visits:
                        assert not scheduler.partial  # TODO: otherwise, model will not be a DTMC
                        dtmc_system_model = self.hmd.system_model.apply_scheduler(scheduler,
                                                                                  drop_unreachable_states=False)
                        assert dtmc_system_model.nr_states == self.hmd.system_model.nr_states
                        expected_number_of_visits = sp.compute_expected_number_of_visits(stormpy.Environment(),
                                                                                         dtmc_system_model)
                        valuation_weights = self._compute_valuation_weigths(expected_number_of_visits)
                        howmany = upto(
                            int((self._options.nlargest_weights_individually_percent + j*0.5) / 100 * len(
                                self.hmd.valuation_id_to_state_ids)),150)
                        important_valuations = heapq.nlargest(howmany,range(len(valuation_weights)), valuation_weights.get)


                    reached_states = sp.get_reachable_states(self.hmd.system_model, self.hmd.system_model.initial_states_as_bitvector,
                                                             system_model_constraint_states, system_model_target_states,
                                                             maximal_steps= self._options.max_reach_steps,
                                                             choice_filter=scheduler.compute_action_support(
                                                                 self.hmd.system_model.nondeterministic_choice_indices))
                    individual_analysis = important_valuations + [self.hmd.state_id_to_valuation_id.get(state) for state in reached_states]
                else:
                    individual_analysis = []

                bound_update = OnDemandValueUpdate(individual_analysis, model_checking_call)

                self._analyse_system_model(bound_update, extract_scheduler=False)
        return self._global_lb, self._global_ub


class SingleOutputCegarChecker(CegarChecker):
    def __init__(self, hmd, options):
        super().__init__(hmd, options)
        self._previous_lb_result = None

    def _analyse_system_model(self, update, extract_scheduler, only_lower_bound=False):
        self._stats.start_analyse_system()
        runcost_model_ub = self.hmd.system_model.get_reward_model(self.hmd.reward_name)
        runcost_model_lb = self.hmd.system_model.get_reward_model(self.hmd.reward_name + "_lb")
        for val_id in update.valuations:
            if val_id is None:
                continue
            cached_result = self._cache.get(val_id)
            if cached_result is None:
                for state_id in self.hmd.valuation_id_to_state_ids[val_id]:
                    nub = update.get_upper_bound(val_id)
                    nlb = update.get_lower_bound(val_id)
                    runcost_model_ub.set_state_reward(state_id, float(nub))
                    runcost_model_lb.set_state_reward(state_id, float(nlb))
                    if nub == nlb:
                        self._cache[val_id] = nub
        # lb_hint = sp.ExplicitModelCheckerHintDouble()
        # if self._previous_lb_result:
        #     lb_hint.set_result_hint(self._previous_lb_result.get_values())
        # if self._previous_lb_scheduler:
        #     lb_hint.set_scheduler_hint(self._previous_lb_scheduler)
        self._stats.start_parametric_system_modelcheck()
        mc_result = sp.check_model_sparse(self.hmd.system_model, self.hmd.system_formula_lb,
                                      extract_scheduler=extract_scheduler)
        self._previous_lb_result = mc_result
        self._stats.end_parametric_system_modelcheck()
        result_lb = mc_result.at(self.hmd.system_model.initial_states[0])
        assert result_lb >= self._global_lb
        self._global_lb = float(result_lb)
        if extract_scheduler:
            scheduler = mc_result.scheduler
        else:
            scheduler = None
        if not only_lower_bound:
            self._stats.start_parametric_system_modelcheck()
            result_ub = sp.model_checking(self.hmd.system_model, self.hmd.system_formula).at(
                self.hmd.system_model.initial_states[0])
            self._stats.end_parametric_system_modelcheck()
            self._global_ub = float(result_ub)
        self._stats.add_new_global_result(self._global_lb, self._global_ub, self._cache)
        self._stats.end_analyse_system()
        logger.debug(f"New bounds are [{self._global_lb}, {self._global_ub}]")
        return scheduler


class DoubleOuputCegarChecker(CegarChecker):
    def __init__(self, hmd, options):
        self._winvars = dict()
        self._parametric_system_model = None
        super().__init__(hmd, options)
        self._env = sp.Environment()
        self._region_as_dict = dict()

        self._region_checker = spp.create_region_checker(self._env, self._parametric_system_model,
                                                         self.hmd.system_formula, False, False)

    def _analyse_system_model(self, update, extract_scheduler, only_lower_bound=False):
        self._stats.start_analyse_system()
        for val_id in self.hmd.valuation_id_to_state_ids.keys():
            if val_id is None:
                continue
            cached_result = self._cache.get(val_id)
            if cached_result is None:
                if val_id in update.valuations:
                    self._region_as_dict[self._winvars[val_id]] = (update.get_lower_bound(val_id), update.get_upper_bound(val_id))
                    #print(self._region_as_dict[self._winvars[val_id]])

        assert len(self._region_as_dict) == len(self._winvars), "Region must be fully specified!"
        preg = spp.ParameterRegion(self._region_as_dict)
        mcres_lb = self._region_checker.get_bound(self._env, preg, False).constant_part()
        self._global_lb = float(mcres_lb)
        if extract_scheduler:
            logger.warning("Not implemented")
            scheduler = None
        else:
            scheduler = None
        if not only_lower_bound:
            mcres_ub = self._region_checker.get_bound(self._env, preg, True).constant_part()
            self._global_ub = float(mcres_ub)
        self._stats.add_new_global_result(self._global_lb, self._global_ub, self._cache)
        self._stats.end_analyse_system()
        logger.debug(f"New bounds are [{self._global_lb}, {self._global_ub}]")
        return scheduler

    def _initialize(self):
        def _substitute_aux(ratfunc, oldvar, newvar):
            assert ratfunc.denominator.is_constant
            newnum = sp.FactorizedPolynomial(ratfunc.numerator.polynomial().substitute({oldvar: sp.Polynomial(newvar)}),
                                             ratfunc.numerator.cache())
            return sp.FactorizedRationalFunction(newnum, ratfunc.denominator)

        new_transition_matrix_builder = sp.ParametricSparseMatrixBuilder(self.hmd.system_model.nr_choices,
                                                                         self.hmd.system_model.nr_states,
                                                                         self.hmd.system_model.transition_matrix.nr_entries,
                                                                         has_custom_row_grouping=True)

        for valid in self.hmd.valuation_id_to_state_ids.keys():
            self._winvars[valid] = pc.core.Variable(f"win{valid}")
        for state_id in range(self.hmd.system_model.nr_states):
            if self.hmd.system_model.get_nr_available_actions(state_id) > 1:
                raise RuntimeError(
                    f"Invalid model, state {state_id} with valuation {self.hmd.system_model.state_valuations.get_json(state_id)} is a subsystem state, but has multiple outgoing actions")
            new_transition_matrix_builder.new_row_group(state_id)
            first_row = self.hmd.system_model.transition_matrix.get_row_group_start(state_id)
            beyond_last_row = self.hmd.system_model.transition_matrix.get_row_group_end(state_id)
            for row in range(first_row, beyond_last_row):
                for entry in self.hmd.system_model.transition_matrix.get_row(row):
                    if self.hmd.system_model_step_states.get(state_id):
                        new_transition_matrix_builder.add_next_value(row, entry.column,
                                                                     _substitute_aux(entry.value(), self.hmd.win_variable,
                                                                                     self._winvars[self.hmd.state_id_to_valuation_id[
                                                                                         state_id]]))
                    else:
                        new_transition_matrix_builder.add_next_value(row, entry.column, entry.value())

        model_components = sp.SparseParametricModelComponents(transition_matrix=new_transition_matrix_builder.build(),
                                                                   state_labeling=self.hmd.system_model.labeling,
                                                                   reward_models=self.hmd.system_model.reward_models,
                                                                   rate_transitions=False)
        if self.hmd.system_model.has_choice_labeling():
            model_components.choice_labeling = self.hmd.system_model.choice_labeling

        self._parametric_system_model = sp.SparseParametricMdp(model_components)

