# LNS
## Large Neighbourhood Search algorithm for the vehicle routing problem with time windows
Written by Benoît Depaire, Lotte Verdonck and Jeroen Corstjens.

## Getting Started
This repository contains a large neighbourhood search implementation, based on Adaptive Large Neighbourhood Search by [Pisinger and Röpke (2007)](https://www.sciencedirect.com/science/article/pii/S0305054805003023).

The adapative elements are implemented, but commented out.

### Usage Examples
Commandline arguments are passed to the Python script using sys.argv.

* When passing multiple algorithm configuration using a csv-file, for example, use


python ./lns/scripts/try_out.py $id $problem_instance $algorithm_instance $seed $determinism_parameter $cooling_rate $start_temp_control_param $noise_parameter $remove_random $remove_worst $remove_related $insert_greedy_parallel $insert_regret_2 $output $solution_file

* When passing a single configuration, pass the proper values in the right order

python ./lns/scripts/try_out.py 1 ./lns/data/Example_Instance.txt 1 720007 10 0.08 0.78 0.35 True  True  False  False True ./output_%d.txt ./solution_%d.txt

### Requirements
Python 2.7.13
