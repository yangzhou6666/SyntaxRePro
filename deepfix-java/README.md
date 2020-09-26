# Activate Virtual Environment

`./init_env.sh`

You may not use the Tsinghua pip mirror.

# Data Generation
## Preprocess
Two datasets are needed:
1. Benchmark
2. Java error pairs

`mkdir data/java_data` to store java data.

There are over 1.7 million java pairs. We only need a small amount of them for mutation. Use the following command to preprose java data.

`python data_processing/preprocess_for_java.py -r path_to_java_pairs -bd path_to_benchmark -n pair_number`

Here you need to specify:
1. path_to_java_path
2. path_to_benchmark
3. pair_number: how much pairs you want to extract (default 10,000). 

It will create a database in: `/data/java_data/java_data.db`. The process takes around 1 hour.


## Training data generation
Use the following command to generate training data.

`python data_processing/java_training_data_generator.py`

*I set min (75) and max (450) lenth here, so it won't mutate all the programs.*