# Activate Virtual Environment

`./init_env.sh`

You may not use the Tsinghua pip mirror.

# Data Generation
## Preprocess
Two datasets are needed:
1. Benchmark
2. Java error pairs

Under `deepfix-java` directory, use `mkdir data/java_data` to store java data.

There are over 1.7 million java pairs. We only need a small amount of them for mutation. Use the following command to preprocess java data.
If it's your first time to generate data, use the following command:

`python data_processing/preprocess_for_java.py -r path_to_java_pairs -bd path_to_benchmark -n pair_number`

Here you need to specify:
1. path_to_java_path
2. path_to_benchmark
3. pair_number: how much pairs you want to extract (default 10,000). 

It will create a database in: `/data/java_data/java_data.db`. The process takes around 1 hour.

If you want to extract more pairs, but not to change benchmark table, use:

`python data_processing/preprocess_for_java.py -r path_to_java_pairs -bd path_to_benchmark -n pair_number -o`

## Training data generation
Use the following command to generate training data.

`python data_processing/java_training_data_generator.py`

*I set min (75) and max (450) length here, so it won't mutate all the programs.*

## Generate evaluation data

`python data_processing/java_test_data_generator.py`

This command will extract errenous programs from benchmark, and convert them into format that Deepfix can take in.

## Train models

`python -O neural_net/train.py data/network_inputs/Deepfix-Java-seed-1189/bin_0 data/checkpoints/Deepfix-Java-seed-1189/bin_0 -v 0.95`

You need to specify path to data, and path to checkpoint directory.


