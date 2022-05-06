# Level-up

This repository includes a prototypical implementation of an abstraction-refinement loop for hierarchical MDPs.
It is based on

- [1] Sebastian Junges and Matthijs Spaan, *Abstraction-Refinement for Hierarchical Probabilistic Models* , CAV 2022. 

#### This Readme

This readme gives an overview of the functionality. 
It also includes some notes how to reproduce the results for [1].
In particular, the readme consists of five parts:
- Getting Level-up describes how to get a docker container.
- Running Level-up describes the use of the tool, its inputs, outputs, etc.
- Benchmarking describes how to run level-up using tools that help to (re)create benchmark results.
- Source-code structure describes the structure of this repo
- Installing Level-up

## Getting Level-up

The easiest way to a working copy of level-up is by getting the docker container.
Details on how to build the docker container manually and/or how to install level-up on your own machine are attached at the bottom of this readme.

#### 1. Load or pull the Docker container
First, either pull the docker container from dockerhub
```
docker pull sjunges/levelup:cav22
```
or, in case you downloaded the docker container:
```
docker load -i levelup.tar
```
The container is based on an container for the probabilistic model checker as provided by the Storm developers, for details, 
see [this documentation](https://www.stormchecker.org/documentation/obtain-storm/docker.html).

#### 2. Boot the container
The following command will run the docker container (for Windows platforms, please see the documentation from the storm website).
```
docker run -w /opt/levelup --rm -it --name levelup sjunges/levelup:cav22
```
If you want to have a shared folder to copy results or files, it is useful to use:
```
docker run --mount type=bind,source="$(pwd)",target=/data -w /opt/levelup --rm -it --name levelup sjunges/levelup:cav22
```
Files that one copies into `/data` are available on the host system in the current working directory. 

You will see a prompt inside the docker container at `/opt/levelup`. 
You can find storm/stormpy including sources at
`/opt/storm` and `/opt/stormpy` respectively. 


## Running Level-up

*Disclaimer*: Level-up is an early-stage academic prototype. It does not check for the validity of the inputs and may crash without helpful error messages.

### Example Invocation
To run level-up, you can use the command-line interface, e.g.,
```
python levelup-cli.py -m CEGAR -i resources/examples/network/protocol.nm -ic M=5 -s resources/examples/network/simplepacket.nm -sc N=10 -map resources/examples/network/vars.map --reward-name time -eta 0.05
```
The inputs are as follows:
- `-m CEGAR` selects CEGAR as a method. The other option is `-m ITERATE`.
- `-i resources/examples/network/protocol.nm` describes the prism program for the uncertain macro-MDP (with open constants).
- `-ic M=5` sets the constants in the above prism program to a fixed value, in this case `M=5`.
- `-s resources/examples/network/simplepacket.nm`desribes the templated subMDP as a prism program (with open constants).
- `-sc N=10` sets the constants in the above prism program.
- `-map resources/examples/network/vars.map` describes a mapping from uncertain-macro MDP states/transitions to the templated subMDPs
- `--reward-name time` the reward model that is used (the reward model occurs in the prism program)
- `-eta` sets the (relative) gap that is acceptable; i.e., the necessary precision of the result. 

The output in this case is fairly minimal and consists of the computed bounds. 
To understand what is going on, consider setting `-v info` or `-v debug`.

Further options are displayed using 
```
python levelup-cli.py --help
```



## Benchmarking
Benchmarking is supported by two scripts: 
One to run levelup in a batch process, and one to collect the stats-files written by the benchmark tool and export them to a table.

### Collecting results

To obtain results for a set of benchmarks, we include a benchmark script. 
To test the script, please use:
```
mkdir test_output && python util/run_benchmarks.py -s TEST -m FULL -o test_output 
```
This test should terminate within few minutes. 
The command line will contain a summary, and the folder test_output will contain logs and stats-files, 
as well as plots visualising the performance of the CEGAR algorithm.

To generate the data for the table, please use:
```
mkdir benchmark_output && python util/run_benchmarks.py -s CAV22 -m FULL -o benchmark_output -to 900
```
Notice that this script will terminate after around 2 hours. 

### Composing stats into a table

To compose the results into a table, we include `util/stats_to_table.py`.
This small script can take the stats included in a folder and creates a table similar to Table 1 in [1]. 

#### General usage
Generally, to run this script, use
```
python util/stats_to_table.py INPUT_DIR output.tex
```
The result is stored in output.tex and can be manually inspected. 
The output of the script gives some help regarding the files it uses and is mainly helpful for debugging.

To render the table into a PDF, use your favourite latex compiler, e.g., use  
```
pdflatex output.tex
```
This creates a file output.pdf.

#### Usage examples
- To reconstruct the table in [1], please run:
```
python util/stats_to_table.py reference_results/ reference_table.tex
```
- To reconstruct the table with your own experiments, please run:
```
python util/stats_to_table.py benchmark_output/ benchmark_table.tex
```

## Source code structure
The most important code is in `levelup/cegar.py` where one can see the implementation of Alg. 1. 
The enumerative baseline that we created is given in `levelup/enumchecker.py`.

We remark that the analysis of parametric models is completely done using the Storm(py) API. 

## Installing from source
To install levelup on your local machine, consider the following:
- Install Storm with Python APIs in [the usual way](https://moves-rwth.github.io/stormpy/installation.html) -- using the `master` branches.
- Run `python setup.py install` or equivalently `pip install .`

The docker container is created by using the Dockerfile. 
You can create this docker container without having storm locally installed. 

