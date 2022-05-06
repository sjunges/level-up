import argparse
import json
import logging
import signal
import time
import os
import sys

import pycarl

sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))
import levelup


def short_str(method):
    if method == levelup.Method.Iterate:
        return "enum"
    if method == levelup.Method.CEGAR:
        return "cegar"
    raise RuntimeError(f"Invalid method {method}")


def run(method, benchmark, acceptable_gap, output_directory):
    hmd = levelup.construct_hierarchical_model_description(benchmark.sysmodel, benchmark.sysconst,
                                                         benchmark.submodel, benchmark.subconst,
                                                         benchmark.mapping, benchmark.rewardmodel)
    checker = levelup.configure_checker(method, hmd, acceptable_gap)
    start_time = time.monotonic()
    result = checker.run()
    total_time = time.monotonic() - start_time
    if hasattr(result, "__len__") and len(result) == 2:
        print(f"\tresult: [{float(result[0])}, {float(result[1])}] in {total_time}s")
    else:
        print(f"\tresult: {float(result)} in {total_time}s")
    with open(os.path.join(output_directory,f"{benchmark.id}.{short_str(method)}stats"), 'w') as statsfile:
        j = json.dumps(checker.stats.to_dict())
        statsfile.write(j)
    return result, total_time, checker.stats


def write_model_stats(benchmark, output_directory):
    with open(os.path.join(output_directory, f"{benchmark.id}.stats"), 'w') as file:
        hmd = levelup.construct_hierarchical_model_description(benchmark.sysmodel, benchmark.sysconst,
                                                             benchmark.submodel, benchmark.subconst,
                                                             benchmark.mapping, benchmark.rewardmodel)
        j = json.dumps(hmd.model_stats)
        file.write(j)


def timeout_handler(signum, frame):
    raise TimeoutError("Timeout")


def check_benchmark(benchmark, config, output_directory):
    pycarl.clear_pools()
    print(f"levelup call: {benchmark.tolevelupcall()}")
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=os.path.join(output_directory,f"{benchmark.id}.log"), encoding='utf-8', level=logging.DEBUG)
    if config.write_model_stats:
        print("\tCollect model stats and write them to file.")
        write_model_stats(benchmark, output_directory)
    print("\tRun abstraction-refinement approach.")
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(config.timeout)
    cegar_success = False
    try:
        _, _, stats = run(levelup.Method.CEGAR, benchmark, config.acceptable_gap, output_directory)
        cegar_success = True
    except TimeoutError:
        print("\tTimeout!")
    signal.alarm(0)
    if config.run_enum:
        print("\tRun enumerative approach.")
        signal.alarm(config.timeout)
        try:
            result, t, _ = run(levelup.Method.Iterate, benchmark, config.acceptable_gap, output_directory)
        except TimeoutError:
            print("\tTimeout!")
            result = None
        signal.alarm(0)
    if config.draw_performance_plot and cegar_success:
        print("\tPlot performance chart.")
        if result is not None:
            stats.plot_timing(benchmark.description, ground_truth=result, baseline_performance=t)
        else:
            stats.plot_timing(benchmark.description)


class Configuration:
    def __init__(self, write_model_stats, run_enum, draw_performance_plot, acceptable_gap=0.05):
        self.write_model_stats = write_model_stats
        self.run_enum = run_enum
        self.draw_performance_plot = draw_performance_plot
        self.acceptable_gap = acceptable_gap
        self.timeout = 1000000000000


class Benchmark:
    def __init__(self, sysmodel, sysconst, submodel, subconst, mapping, rewardmodel, id, description):
        self.sysmodel = sysmodel
        self.sysconst = sysconst
        self.submodel = submodel
        self.subconst = subconst
        self.mapping = mapping
        self.rewardmodel = rewardmodel
        self.id = id
        self.description = description

    def tolevelupcall(self):
        return "-i " + self.sysmodel + " -ic " + self.sysconst + " -s " + self.submodel + " -sc " + self.subconst + " -map " + self.mapping + " -reward-name " + self.rewardmodel


CAV22_MAIL_BENCHMARKS = [Benchmark("resources/examples/simple_mail/system.nm", "HORIZON=10",
                                   "resources/examples/simple_mail/step.nm", "",
                                   "resources/examples/simple_mail/vars.map", "runcost",
                                   id="mail-10", description="mail with horizon 10"),
                         Benchmark("resources/examples/simple_mail/system.nm", "HORIZON=12",
                                  "resources/examples/simple_mail/step.nm", "",
                                  "resources/examples/simple_mail/vars.map", "runcost",
                                  id="mail-12", description="mail with horizon 12")]
CAV22_NETW_BENCHMARKS = [Benchmark("resources/examples/network/protocol.nm", "M=30",
                                   "resources/examples/network/simplepacket.nm", "N=50",
                                   "resources/examples/network/vars.map", "time",
                                   id="netw-30-50", description="network-30-50"),
                        Benchmark("resources/examples/network/protocol.nm", "M=30",
                                   "resources/examples/network/simplepacket.nm", "N=80",
                                   "resources/examples/network/vars.map", "time",
                                   id="netw-30-80", description="network-30-80"),
                        Benchmark("resources/examples/network/protocol.nm", "M=50",
                                   "resources/examples/network/simplepacket.nm", "N=80",
                                   "resources/examples/network/vars.map", "time",
                                   id="netw-50-80", description="network-50-80")]
CAV22_CORR_BENCHMARKS = [Benchmark("resources/examples/corridor/floor.nm", "M=11,Rms=10",
                                   "resources/examples/corridor/room.nm", "N=50",
                                    "resources/examples/corridor/vars.map", "time",
                                   id="corr-11-10-50", description="corridor-11-10-room-50"),
                         Benchmark("resources/examples/corridor/floor.nm", "M=11,Rms=8",
                                   "resources/examples/corridor/room.nm", "N=100",
                                   "resources/examples/corridor/vars.map", "time",
                                   id="corr-11-8-100", description="corridor-11-8-room-100"),
                         Benchmark("resources/examples/corridor/floor.nm", "M=11,Rms=8",
                                   "resources/examples/corridor/room.nm", "N=200",
                                   "resources/examples/corridor/vars.map", "time",
                                   id="corr-11-8-200", description="corridor-11-8-room-200"),
                         Benchmark("resources/examples/corridor/floor.nm", "M=13,Rms=11",
                                   "resources/examples/corridor/room.nm", "N=50",
                                   "resources/examples/corridor/vars.map", "time",
                                   id="corr-13-11-50", description="corridor-11-10-room-50")]
CAV22_CORR1_BENCHMARKS = [Benchmark("resources/examples/corridor/floor_1d.nm", "M=17,Rms=14",
                                   "resources/examples/corridor/room.nm", "N=75",
                                   "resources/examples/corridor/vars.map", "time",
                                   id="corr1-17-14-75", description="corridor1-17-14-room-75"),
                          Benchmark("resources/examples/corridor/floor_1d.nm", "M=18,Rms=15",
                                  "resources/examples/corridor/room.nm", "N=75",
                                  "resources/examples/corridor/vars.map", "time",
                                  id="corr1-18-15-75", description="corridor1-18-15-room-75"),
                          Benchmark("resources/examples/corridor/floor_1d.nm", "M=25,Rms=20",
                                  "resources/examples/corridor/room.nm", "N=75",
                                  "resources/examples/corridor/vars.map", "time",
                                  id="corr1-25-20-75", description="corridor1-25-20-room-75")]
CAV22_SDN_BENCHMARKS = [Benchmark("resources/examples/sdn/conditions.nm", "HORIZON=5",
                                  "resources/examples/sdn/routing.nm", "N=12,Z=4,M=4",
                                  "resources/examples/sdn/vars.map", "time",
                                  id="sdn-5-12-4-4", description="sdn-5-12-4-4"),
                        Benchmark("resources/examples/sdn/conditions.nm", "HORIZON=5",
                                  "resources/examples/sdn/routing.nm", "N=8,Z=4,M=4",
                                  "resources/examples/sdn/vars.map", "time",
                                  id="sdn-5-8-4-4", description="sdn-5-8-4-4"),
                        Benchmark("resources/examples/sdn/conditions.nm", "HORIZON=6",
                                  "resources/examples/sdn/routing.nm", "N=8,Z=4,M=4",
                                  "resources/examples/sdn/vars.map", "time",
                                  id="sdn-6-8-4-4", description="sdn-6-8-4-4")]

TEST_BENCHMARKS = [Benchmark("resources/examples/corridor/floor.nm", "M=6,Rms=5",
                                   "resources/examples/corridor/room.nm", "N=20",
                                    "resources/examples/corridor/vars.map", "time",
                                   id="corr-6-5-20", description="corridor-6-5-room-20"),
                   Benchmark("resources/examples/simple_mail/system.nm", "HORIZON=3",
                                   "resources/examples/simple_mail/step.nm", "",
                                   "resources/examples/simple_mail/vars.map", "runcost",
                                   id="mail-3", description="mail with horizon 3"),
                   Benchmark("resources/examples/network/protocol.nm", "M=5",
                                   "resources/examples/network/simplepacket.nm", "N=10",
                                   "resources/examples/network/vars.map", "time",
                                   id="netw-5-10", description="network-5-10"), ]

CAV22_BENCHMARKS = CAV22_NETW_BENCHMARKS + CAV22_CORR_BENCHMARKS + CAV22_CORR1_BENCHMARKS + CAV22_MAIL_BENCHMARKS + CAV22_SDN_BENCHMARKS

RUN_ALL_CONFIGURATION = Configuration(write_model_stats=True, run_enum=True, draw_performance_plot=True)
RUN_ONLY_CEGAR = Configuration(write_model_stats=False, run_enum=False, draw_performance_plot=False)


def main():
    parser = argparse.ArgumentParser(description='Run benchmarks for levelup')
    parser.add_argument('--set', '-s', help="Set to use", choices=["TEST", "CAV22"], default="CAV22")
    parser.add_argument('--mode', '-m', help="Mode to use", choices=["ONLYCEGAR", "FULL"], default="FULL")
    parser.add_argument('--output', '-o', help="Path for output", default="benchmark_results")
    parser.add_argument('--timeout', '-to', help="Timeout in seconds", type=int, default=15*60)
    args = parser.parse_args()
    if args.set == "TEST":
        benchmarkset = TEST_BENCHMARKS
    elif args.set == "CAV22":
        benchmarkset = CAV22_BENCHMARKS
    else:
        assert False, "invalid set"

    if args.mode == "ONLYCEGAR":
        config = RUN_ONLY_CEGAR
    elif args.mode == "FULL":
        config = RUN_ALL_CONFIGURATION
    config.timeout = args.timeout

    if not os.path.isdir(args.output):
        raise RuntimeError(f"{args.output} is not an existing directory.")

    for nr, benchmark in enumerate(benchmarkset):
        print(f"Running {nr+1}/{len(benchmarkset)}: {benchmark.description}...")
        check_benchmark(benchmark, config, args.output)


if __name__ == "__main__":
    main()