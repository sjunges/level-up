import argparse
import levelup
import logging

logger = logging.getLogger(__name__)

def method_from_string(method_str):
    if method_str == "CEGAR":
        return levelup.Method.CEGAR
    elif method_str == "ITERATE":
        return levelup.Method.Iterate
    else:
        raise RuntimeError(f"Invalid method string {method_str}")


def main():
    parser = argparse.ArgumentParser(description='Level-up. A prototype for verifying hierarchical MDPs.')
    parser.add_argument('--method', '-m', choices=["CEGAR", "ITERATE"], help="Method to use", default="CEGAR")
    parser.add_argument('--system-model', '-i', help='System model file', required=True)
    parser.add_argument('--system-constants', '-ic', help="System model constant values", default="")
    parser.add_argument('--step-model', '-s', help='Step model file', required=True)
    parser.add_argument('--step-constants', '-sc', help="Step model constants", default="")
    parser.add_argument('--varmap', '-map', help="map from system variables to step constants", required=True)
    parser.add_argument('--reward-name', help="reward model", required=True)
    parser.add_argument('--acceptable-gap', '-eta', type=float, help="Acceptable gap/required precision", default=0.05)
    parser.add_argument('--verbosity', choices=["debug", "info", "resultonly", "silent"], default="resultonly")
    args = parser.parse_args()
    if args.verbosity == "debug":
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    elif args.verbosity == "info":
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    else:
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.ERROR)

    hmd = levelup.construct_hierarchical_model_description(args.system_model, args.system_constants, args.step_model,
                                                           args.step_constants, args.varmap, args.reward_name)
    if args.acceptable_gap < 0 or args.acceptable_gap > 1:
        raise RuntimeError("Acceptable gap must be between 0 and 1.")
    checker = levelup.configure_checker(method_from_string(args.method), hmd, args.acceptable_gap)
    result = checker.run()
    if args.verbosity == "resultonly":
        print(f"Final result is {result}.")
    else:
        logger.info(f"Final result is {result}.")

if __name__ == "__main__":
    main()
