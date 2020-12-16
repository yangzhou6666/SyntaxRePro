# First, generate training_data
echo 'Preprocessing Blackbox dataset...'
python data_processing/java_data_preprocess.py

echo 'Generating training and validation dataset...'
python -O data_processing/java_training_data_generator.py


# Make sure that you have test_raw data under Deepfix dir
echo 'Converting DeepFix-Java dataset to RLAssist-Java format...'
python -O data_processing/java_deepfix_to_rlassist_test_data_convertor.py

# 准备进行训练
python -O neural_net/agent.py data/network_inputs/RLAssist-seed-1189/bin_0/ data/checkpoints/RLAssist-seed-1189/bin_0/

# 进行evaluation
# 注意这里的checkpoint需要自己指定
python -O neural_net/agent.py data/network_inputs/RLAssist-seed-1189/bin_0/ data/checkpoints/RLAssist-seed-1189/bin_0/ -eval 30342250 -wh real -w 4