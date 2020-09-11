# 需要先开启虚拟环境

# 首先对Java数据进行预处理，将一部分Java数据写入数据库
# 在这里会创建两张数据表，分别是Code和Benchmark
python data_processing/preprocess_for_java.py 


# 在data/checkpoints/Deepfix-Java-seed-1189/路径下生成training data
python data_processing/java_training_data_generator.py

# 从Benchmark数据表中提取数据，并生成用于最终evaluation的真实数据
# 要注意一般并不会提取所有的数据，因为有长度范围的限制
python data_processing/java_test_data_generator.py

# 进行训练
python -O neural_net/train.py data/network_inputs/Deepfix-Java-seed-1189/bin_1 data/checkpoints/Deepfix-Java-seed-1189/bin_1 -v 0.95

# 进行evaluation 指定checkpoint的路径
python -O post_processing/generate_java_fixes.py data/checkpoints/Deepfix-Java-seed-1189/bin_1 -v 0.6