binid=4
CUDA_VISIBLE_DEVICES=1 python -O neural_net/train.py data/network_inputs/iitk-typo-1189/bin_"$binid"/ data/checkpoints/iitk-typo-1189/bin_"$binid"/
CUDA_VISIBLE_DEVICES=1 python -O neural_net/train.py data/network_inputs/iitk-ids-1189/bin_"$binid"/ data/checkpoints/iitk-ids-1189/bin_"$binid"/