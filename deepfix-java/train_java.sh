# Activate virtual environment first

# create a direction to store data
mkdir data/java_data

# First we preporcess Java data, store part of them into a sqlite3 database.
# In the database, we create two tables, Code and Benchmark
python data_processing/preprocess_for_java.py 

# in data/checkpoints/Deepfix-Java-seed-1189/ we generate training data
python data_processing/java_training_data_generator.py

python data_processing/java_test_data_generator.py

python -O neural_net/train.py data/network_inputs/Deepfix-Java-seed-1189/bin_0 data/checkpoints/Deepfix-Java-seed-1189/bin_0 -v 0.95

python -O post_processing/generate_java_fixes.py data/checkpoints/Deepfix-Java-seed-1189/bin_0 -v 0.95
# Do evaluation, speicify path to checkpoint: data/checkpoints/Deepfix-Java-seed-1189/bin_0