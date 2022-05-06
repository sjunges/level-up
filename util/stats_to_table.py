import json
import math
import sys
from pathlib import Path

TEXHEADER = """
\\documentclass{article}
\\usepackage{amsmath}
\\usepackage{amssymb}
\\usepackage{adjustbox}
\\usepackage[a4paper, landscape, margin=0.6in]{geometry}

\\title{Benchmark Result Report}
\\author{Level-Up}


\\newcommand{\\mdp}{\\mathcal{M}}
\\newcommand{\\classmdp}{\\mathcal{T}}
\\newcommand{\\ntparti}{\\mathbb{I}}
\\newcommand{\\macro}[1]{\\mu(#1)}

\\begin{document}
\\maketitle{}

The table below contains the results.

\\begin{table}[h!]
\\centering
\\caption{Benchmark statistics, runtimes of the approaches, and details from levelup}
\\scalebox{0.82}{
\\begin{tabular}{lcrrrrrrr||r|rrr||rrrrr}
Name & Inst & $|S_\\mdp|$ & $|\\ntparti|$ &
$|S_{\\macro{\\mdp}}|$ & $|\\text{A}_{\\macro{\\mdp}}|$   &
$|S_\classmdp|$ & $|\\text{A}_\classmdp|$ & 
$t_\\text{init}$ & $t_\\text{enum}$ & $t_{50}$ & $t_{90}$ & $t_{95}$ & iter. & indrf. & $\\stackrel{\\text{um}}{\\%}$ & $\\stackrel{\\text{sr}}{\\%}$  & $\\stackrel{\\text{ir}}{\\%}$ 
\\\\\\hline
"""

TEXFOOTER = """
\\end{tabular}
}
\\end{table}
\\end{document}
"""


def to_order_of_mag(n):
    return "10^\\textbf{" + str(round(math.log10(n))) + "}"


def create_tex_file(input_dir, output_file):
    with open(output_file, 'w') as outfile:
        outfile.write(TEXHEADER)
    pathlist = Path(input_dir).glob('**/*.stats')
    for path in sorted(pathlist):
        print(f"Opening {path}")
        with open(path) as infile:
            mdata = json.load(infile)
            filename = path.stem
            print(f"filename: {filename}")
            family = filename.split("-")[0]
            print(f"\tfamily: {family}")
            inst = ",".join(filename.split("-")[1:])
        row_raw = ["\\textsf{" + family + "}", "\\tiny{" + inst + "}", to_order_of_mag(mdata["subMDP-states"] * mdata["nr-valuations"] + mdata["macroMDP-states"]), mdata["nr-valuations"], mdata["macroMDP-states"], mdata["macroMDP-choices"],
                   mdata["subMDP-states"],
                   mdata["subMDP-choices"], round(mdata["startup-time"]) if mdata["startup-time"] >= 1 else "<1"]
        enumpath = path.with_suffix(".enumstats")
        if enumpath.exists():
            print("\tfound enumeration stats")
            with open(enumpath) as enumfile:
                enumdata = json.load(enumfile)

            total_enum_time = float(enumdata["system mc time"]) + float(enumdata["instantiations time"])
            row_raw += []
        else:
            row_raw += []
        row1 = [str(x) for x in row_raw] # + ["$" + str(x) + "$" for x in row_raw[2:]]

        cegarpath = path.with_suffix(".cegarstats")
        if cegarpath.exists():
            print("\tfound cegar stats")
            with open(cegarpath) as cegarfile:
                cegardata = json.load(cegarfile)
            if enumpath.exists():
                timing = cegardata["timing"]

            row3_raw = row1 + [round(total_enum_time), round(cegardata["p50-timing"]),
                       round(cegardata["p90-timing"]), round(timing),
                       cegardata["nr_subsystem_analyses"],  cegardata["nr_subsystem_samples"],
                        round(100*float(cegardata["system_parametric_mc_time"])/timing),
                        round(100*float(cegardata["subsystem_set_analysis_time"])/timing),
                        round(100*float(cegardata["sample_subsystem_time"])/timing)]
            row3 = [str(x) for x in row3_raw[:2]] + ["$" + str(x) + "$" for x in row3_raw[2:]]
            with open(output_file, 'a') as outfile:
                outfile.write(" &\t".join(row3))
                outfile.write("\\\\\n")
    with open(output_file, "a") as outfile:
        outfile.write(TEXFOOTER)


if __name__ == "__main__":
    print("Running stats_to_table.py.")
    print("Usage: stats_to_table.py INPUT_DIR output_file")
    if len(sys.argv) != 3:
        raise RuntimeError("Expected input dir and output file")
    print(f"INPUT_DIR: {sys.argv[1]}, output_file: {sys.argv[2]}")
    create_tex_file(sys.argv[1], sys.argv[2])


