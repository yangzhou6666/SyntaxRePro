#encoding:  utf-8

import os
import sqlite3
import json
from functools import partial
import numpy as np
import javac_parser
import argparse


def generate_raw_test_data(db_path, bins,  min_program_length, max_program_length):
    '''
    打算从benchmark生成数据，而并让training data和ttest data用同一个数据集
    然后我们并不需要bins这一说，我们只希望
    我们把文件名直接拿来作为code_id
    '''
    raw_test_data = {}
    program_lengths = []

    # 同样的，这里的bins是为了得到raw data的problem_id，然后才能从数据库中取出
    # 而我们是不需要的，我们直接取出所有的
    problem_list = []
    for bin_ in bins:
        for problem_id in bin_:
            problem_list.append(problem_id)

    skipped = 0
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        query = "SELECT code_id, tokenized_code, name_dict, name_seq FROM Benchmark " +\
            "WHERE codelength>? and codelength<?;"
        for row in cursor.execute(query, (min_program_length, max_program_length)):
            code_id, tokenized_code = map(str, row[:-2])
            name_dict, name_sequence = json.loads(
                row[2]), json.loads(row[3])

            program_length = len(tokenized_code.split())
            program_lengths.append(program_length)

            # 在我们的程序中，problem_id就是code_id
            problem_id = code_id

            if program_length >= min_program_length and program_length <= max_program_length:
                try:
                    raw_test_data[problem_id].append(
                        (tokenized_code, name_dict, name_sequence, code_id))
                except KeyError:
                    raw_test_data[problem_id] = [
                        (tokenized_code, name_dict, name_sequence, code_id)]
            else:
                skipped += 1
                print 'out of length range:', code_id


    print 'problem_count:', len(raw_test_data)
    print 'program_count:', sum([len(raw_test_data[problem_id]) for problem_id in raw_test_data])
    print 'discared_problems:', skipped

    program_lengths = np.sort(program_lengths)

    print 'Statistics'
    print '----------'
    print 'Program length:  Mean =', np.mean(program_lengths), '\t95th %ile =', \
        program_lengths[int(0.95 * len(program_lengths))]

    return raw_test_data

def save_bins(destination, test_data, bins, which='raw'):
    print which
    full_list = []

    for i in range(5):
        np.save(os.path.join(destination, 'bin_%d' % i, 'test_%s_bin_%d.npy' % (which, i)), test_data)


if __name__ == '__main__':
    max_program_length = 45000
    min_program_length = 1
    max_fix_length = 25
    max_mutations = 5
    programs_per_problem = 100
    seed = 1189

    db_path 		 = 'data/java_data/java_data.db'
    output_directory = os.path.join('data', 'network_inputs', "Deepfix-Java-seed-%d" % (seed,))

    # mock bins
    bins = {}

    assert os.path.exists(output_directory)

    raw_test_data = generate_raw_test_data(
        db_path, bins, min_program_length, max_program_length)
    save_bins(output_directory, raw_test_data, bins, 'raw')