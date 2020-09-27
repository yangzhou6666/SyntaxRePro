# First, generate training_data
# 这一步是和Deepfix一样的，我现在不想重构代码，就在RLAssist路径下也再生成一次把
echo 'Preprocessing Blackbox dataset...'
python data_processing/java_data_preprocess.py

echo 'Generating training and validation dataset...'
python -O data_processing/java_training_data_generator.py


# 在运行这一步之前确保Deepfix目录下已经生成了test_raw文件
echo 'Converting DeepFix-Java dataset to RLAssist-Java format...'
python -O data_processing/java_deepfix_to_rlassist_test_data_convertor.py

# 准备进行训练
python -O neural_net/agent.py data/network_inputs/RLAssist-seed-1189/bin_0/ data/checkpoints/RLAssist-seed-1189/bin_0/

# 进行evaluation
# 注意这里的checkpoint需要自己指定
python -O neural_net/agent.py data/network_inputs/RLAssist-seed-1189/bin_4/ data/checkpoints/RLAssist-seed-1189/bin_4/ -eval 21063552 -wh real -w 4