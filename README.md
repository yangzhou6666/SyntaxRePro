# Data preprocess

First cd to the root directory.

Two datasets are needed:
1. Benchmark
2. Java error pairs

`mkdir java_data` to store java data.

There are over 1.7 million java pairs. We only need a small amount of them for mutation. Use the following command to preprose java data.
If it's your first time to generate data, use the following command:

`python preprocess_for_java.py -r path_to_java_pairs -bd path_to_benchmark -n pair_number`

Here you need to specify:
1. path_to_java_path
2. path_to_benchmark
3. pair_number: how much pairs you want to extract (default 10,000). 

It will create a database in: `/data/java_data/java_data.db`. The process takes around 1 hour.

If you want to extract more pairs, but not to change benchmark table, use:

`python preprocess_for_java.py -r path_to_java_pairs -bd path_to_benchmark -n pair_number -o`

# Deepfix Experiment

## Create and activate a virtual environment

cd to `deepfix-java` directory, then

`./init_env.sh`

You may not use the Tsinghua pip mirror.

## Data Generation
Switch to `deepfix-java` directory.

### Training data generation
Use the following command to generate training data.

`python data_processing/java_training_data_generator.py`

*I set min (75) and max (450) length here, so it won't mutate all the programs.*

### Generate evaluation data

`python data_processing/java_test_data_generator.py`

This command will extract errenous programs from benchmark, and convert them into format that Deepfix can take in.

# RLAssist Experiment
## Create and activate a virtual environment

cd to `rlasssit-java` directory, then

`./init_env.sh`

## Data Generation
Switch to `rlassist-java` directory.

### Training data generation

Use the following command to generate training data.

`python data_processing/java_training_data_generator.py`

### Convert Deepfix evaluation data to RLAssist format

`python -O data_processing/java_deepfix_to_rlassist_test_data_convertor.py`

