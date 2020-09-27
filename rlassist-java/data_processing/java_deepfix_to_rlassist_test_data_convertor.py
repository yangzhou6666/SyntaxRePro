#encoding:  utf-8
import os
import argparse
import sqlite3
import numpy as np
import json

from data_processing.training_data_generator import load_dictionaries
from util.helpers import remove_empty_new_lines
from util.c_tokenizer import C_Tokenizer

deepfix_base_dir = '../deepfix-java/data/network_inputs/Deepfix-Java-seed-1189' 
RLAssist_base_dir = 'data/network_inputs/RLAssist-seed-1189/'

max_program_len = 450

dummy_correct_program = 'EOF -new-line- _pad_'

tokenize = C_Tokenizer().tokenize
convert_to_new_line_format = C_Tokenizer().convert_to_new_line_format
convert_to_rla_format = lambda x: remove_empty_new_lines(convert_to_new_line_format(x))

raw_test_data = {}

def generate_name_dict_store(db_path, bins):
    '''
    给定数据库的路径
    将数据库中每一个tokenzied_code所对应的name_dict和name_seq存储到一个dictionary中
    '''
    name_dict_store={}
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        for row in cursor.execute("SELECT code_id, name_dict, name_seq FROM Benchmark;"):
            code_id = str(row[0])
            name_dict = json.loads(row[1])
            name_seq = json.loads(row[2])
            name_dict_store[code_id] = (name_dict, name_seq)
    print 'name_dict_store len:', len(name_dict_store)
    
    return name_dict_store

def vectorize(tokens, tldict, max_vector_length=max_program_len):
    vec_tokens = []
    for token in tokens.split():
        try:
            vec_tokens.append(tldict[token])
        except Exception:
            return None

    if len(vec_tokens) > max_vector_length:
        return None

    return vec_tokens

# convert the df test data into rla format
n = 1
for bin_id in range(n):
    raw_test_data[bin_id] = {}

    bin_raw_test_data = np.load(os.path.join(deepfix_base_dir, 'bin_%d' % bin_id, 'test_raw_bin_%d.npy' % bin_id), allow_pickle=True).item()

    # 针对raw_test_data
    for problem_id, test_programs_ in bin_raw_test_data.iteritems():
        raw_test_data[bin_id][problem_id] = []
        for incorrect_program, name_dict, name_sequence, code_id in test_programs_:
            raw_test_data[bin_id][problem_id].append((code_id, convert_to_rla_format(incorrect_program), dummy_correct_program))

skipped = 0

db_path = '../java_data/java_data.db'
name_dict_store = generate_name_dict_store(db_path, {})


for bin_id in range(n):
    print 'bin_%d' % bin_id,
    target_bin_dir = os.path.join(RLAssist_base_dir, 'bin_%d' % bin_id)
    tl_dict, _ = load_dictionaries(target_bin_dir)

    for which, test_data in [('raw', raw_test_data)]:
        test_data_this_fold = {}
        for problem_id in test_data[bin_id]:
            for code_id, inc_tokens, cor_tokens in test_data[bin_id][problem_id]:
                inc_vector = vectorize(inc_tokens, tl_dict)
                corr_vector = vectorize(cor_tokens, tl_dict)

                if inc_vector is None or corr_vector is None:
                    skipped += 1
                    continue

                test_data_this_fold[code_id] = (inc_vector, corr_vector)
        print which, len(test_data_this_fold),
        np.save(os.path.join(target_bin_dir, 'test_%s.npy' % which), test_data_this_fold)
        
        print os.path.join(target_bin_dir, 'test_name_dict_store.npy')
        np.save(os.path.join(target_bin_dir, 'test_name_dict_store.npy'), name_dict_store)
    print
