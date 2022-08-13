import enum
import logging

from levelup.model import HierarchicalModelDescription
from levelup.enumchecker import SingleOutputFullInstantiationChecker, BinaryOutputFullInstantiationChecker
from levelup.cegar import SingleOutputCegarChecker, DoubleOuputCegarChecker, CegarCheckerOptions

import stormpy as sp

logger = logging.getLogger("levelup")


def parse_variable_mapping(sys_prog, step_prog, map_file):
    """

    :return:
    """
    logger.debug("Parse variable mapping...")
    map = dict()

    sysvars = dict()
    for v in sys_prog.global_integer_variables:
        sysvars[v.name] = v.expression_variable
    stepconstants = dict()
    for c in step_prog.constants:
        stepconstants[c.name] = c.expression_variable

    with open(map_file) as file:
        for line in file:
            if line.startswith("//"):
                continue
            entry = line.strip().split()
            sysvar = sysvars.get(entry[0])
            if sysvar is None:
                raise ValueError(f"Do not know variable {entry[0]}")
            stepconstant = stepconstants.get(entry[1])
            if stepconstant is None:
                raise ValueError(f"Do not know constant {entry[1]}")
            map[sysvar] = stepconstant
    logger.debug("...done parsing variable mapping")
    return map


class Method(enum.Enum):
    Iterate = 1
    CEGAR = 2


def construct_hierarchical_model_description(system_file, system_constants, step_file, step_constants, variable_map_file, reward_name):
    logger.debug("Prepare system model...")
    logger.debug(f"Use {system_file}...")
    system_program = sp.parse_prism_program(system_file)
    system_properties = sp.parse_properties('R{"' + reward_name + '"}min=? [F "done"]', system_program)
    system_program, system_properties = sp.preprocess_symbolic_input(system_program,
                                                                     system_properties, system_constants)
    system_formula = system_properties[0].raw_formula
    system_program = system_program.as_prism_program()
    if not system_program.has_label("done"):
        raise RuntimeError("System program must have a done label")
    system_build_options = sp.BuilderOptions([system_formula])
    system_build_options.set_build_state_valuations()
    system_build_options.set_build_all_labels()
    logger.info("Building system model...")
    if system_program.has_constant("win"):
        system_model = sp.build_sparse_parametric_model_with_options(system_program, system_build_options)
        system_formula_lb = None
    else:
        system_model = sp.build_sparse_model_with_options(system_program, system_build_options)
        system_model.add_reward_model(reward_name + "_lb", system_model.get_reward_model(reward_name))
        system_formula_lb = sp.parse_properties('R{"' + reward_name + '_lb"}min=? [F "done"]', system_program)[
            0].raw_formula
    logger.debug(system_model)

    logger.debug("Prepare step model...")
    logger.debug(f"Use {step_file}...")
    step_program = sp.parse_prism_program(step_file)
    if system_program.has_constant("win"):
        step_properties = sp.parse_properties("Pmax=? [F \"win\"]", step_program)
    else:
        step_properties = sp.parse_properties("Rmin=? [F \"done\"]", step_program)
    step_program, step_properties = sp.preprocess_symbolic_input(step_program, step_properties, step_constants)
    step_formula = step_properties[0].raw_formula
    step_program = step_program.as_prism_program()
    if system_program.has_constant("win"):
        if not step_program.has_label("win"):
            raise RuntimeError("Step program for programs with two outcomes must have a win label")
    else:
        if not step_program.has_label("done"):
            raise RuntimeError("Step program must have a done label")
    step_build_options = sp.BuilderOptions()
    logger.info("Building step model...")
    step_parametric_model = sp.build_sparse_parametric_model_with_options(step_program, step_build_options)
    logger.debug(step_parametric_model)
    system_variables_to_step_constants = parse_variable_mapping(system_program, step_program, variable_map_file)

    return HierarchicalModelDescription(system_model, step_parametric_model, system_variables_to_step_constants,
                                        reward_name, system_formula, step_formula, system_formula_lb)


def configure_checker(method, hmd, acceptable_gap):
    if method == Method.Iterate:
        #1: Relatively naive approach.
        if hmd.has_single_outputs:
            return SingleOutputFullInstantiationChecker(hmd)
        else:
            return BinaryOutputFullInstantiationChecker(hmd)

    elif method == Method.CEGAR:
        options = CegarCheckerOptions(max_reach_steps=2, reassesement_iterations=8, compute_expected_visits=True, acceptable_gap=acceptable_gap)
        if hmd.has_single_outputs:
            cechecker = SingleOutputCegarChecker(hmd, options)
        else:
            cechecker = DoubleOuputCegarChecker(hmd, options)
        return cechecker

