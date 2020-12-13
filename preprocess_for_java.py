#encoding:  utf-8
import os
import sqlite3
import json
import sys
import javac_parser
import argparse
import random

reload(sys)
sys.setdefaultencoding('utf8')

def initialize_database(db_path, raw_data_path, data_length, is_shuffle=False):
    # Initial table
    with sqlite3.connect(db_path) as conn:
        conn.execute('''DROP TABLE IF EXISTS Code''')
        conn.execute('''CREATE TABLE Code (code_id, code text);''')
    
    # Add some new columns
    with sqlite3.connect(db_path) as conn:
        conn.execute('''ALTER TABLE Code ADD tokenized_code text;''')
        conn.execute('''ALTER TABLE Code ADD name_dict;''')
        conn.execute('''ALTER TABLE Code ADD name_seq;''')
        conn.execute('''ALTER TABLE Code ADD codelength integer;''')
    
    # count: data that has been inserted
    count = 0
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        file_list = os.listdir(raw_data_path)
        if is_shuffle:
            random.shuffle(file_list)
        for file_event_id in file_list:
            if count % 10000 == 0:
                print ('%d programs inserted ... ' % (count))
            if count == data_length:
                break
            with open(os.path.join(raw_data_path, file_event_id, '1')) as f:
                code = f.read()
                tokenized_code, name_dict, name_seq = get_tokenized_code(code)
                codelength = len(tokenized_code.split())
                cursor.execute('''INSERT INTO Code (code_id, code, tokenized_code, name_dict, name_seq, codelength) VALUES (?, ?, ?, ?, ?, ?);''', (file_event_id, unicode(code), unicode(tokenized_code), json.dumps(name_dict), json.dumps(name_seq), codelength))
                count += 1    
    

def initialize_Benchmark_table(db_path, benchmark_dir):
    '''
    takes path to database and the path to benchmark
    stores benchmark to the database
    '''
    with sqlite3.connect(db_path) as conn:
        conn.execute('''DROP TABLE IF EXISTS Benchmark''')
        conn.execute('''CREATE TABLE Benchmark (code_id, code text);''')

    with sqlite3.connect(db_path) as conn:
        conn.execute('''ALTER TABLE Benchmark ADD tokenized_code text;''')
        conn.execute('''ALTER TABLE Benchmark ADD name_dict;''')
        conn.execute('''ALTER TABLE Benchmark ADD name_seq;''')
        conn.execute('''ALTER TABLE Benchmark ADD codelength integer;''')
    

    count = 0
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        for file_event_id in os.listdir(benchmark_dir):
            if count % 10000 == 0:
                print ('%d programs from Benchmark inserted ... ' % (count))
            with open(os.path.join(benchmark_dir, file_event_id)) as f:
                code = f.read()
                tokenized_code, name_dict, name_seq = get_tokenized_code(code)
                codelength = len(tokenized_code.split())
                cursor.execute('''INSERT INTO Benchmark (code_id, code, tokenized_code, name_dict, name_seq, codelength) VALUES (?, ?, ?, ?, ?, ?);''', (file_event_id + 'bench', unicode(code), unicode(tokenized_code), json.dumps(name_dict), json.dumps(name_seq), codelength))
                count += 1 
    
def get_tokenized_code(code):
    tokens = []
    name_dict = {}
    name_seq = []
    loc = len(code.split('\n'))
    tokens_by_line = [[i] for i in range(loc)]
    for line in tokens_by_line:
        line.append('~')

    lex_res = java.lex(code)
    for lex in lex_res:
        line_num = lex[2][0]
        if lex[0] == 'IDENTIFIER':
            id = lex[1]
            name_seq.append(id)
            if id in name_dict:
                tokens_by_line[line_num - 1].append("IDENTIFIER" + str(name_dict[id]) + "@")
            else:
                name_dict[id] = len(name_dict) + 1
                tokens_by_line[line_num - 1].append("IDENTIFIER" + str(name_dict[id]) + "@")
        else: 
            tokens_by_line[line_num - 1].append(lex[0])

    # remove blank line
    tokens_by_line_no_blank_lines = []
    for line in tokens_by_line:
        if line[-1] == '~':
            continue
        tokens_by_line_no_blank_lines.append(line)

    # update line number
    for i, line in enumerate(tokens_by_line_no_blank_lines):
        line_num = i
        line[0] = ' '.join(list(str(line_num)))


    token_string_by_line = []
    for line in tokens_by_line_no_blank_lines:
        tokens.append(' '.join(line))
        token_string_by_line.append(' '.join(line))

    return ' '.join(token_string_by_line), name_dict, name_seq


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Preprocess Java programs')
    parser.add_argument('-d', '--db_path', help='database path to store', 
                        default = 'java_data/java_data.db')
    parser.add_argument('-r', '--raw_data_path', help='raw java data path to preprocess',
                        default = '../../java_data/')
    parser.add_argument('-n', '--num_data', type=int, help='Number to data example to generate',
                        default = 10000)
    parser.add_argument('-bd', '--benchmark_dir', help = 'Path to src_files directory',
                        default = '../../src_files/')
    parser.add_argument('-o', '--only_raw_data', help = 'Only update the Code table, do not change Benchmark table',
                        action='store_true', default=False)
    parser.add_argument('-is', '--is_shuffle', help = 'Shuffle the files before extracting data',
                        action='store_true', default=False)

    args = parser.parse_args()
    print '----------------------------------------------------'
    print 'db_path            : ', args.db_path
    print 'raw_data_path      : ', args.raw_data_path
    print 'num_data           : ', args.num_data
    print 'benchmark_dir      : ', args.benchmark_dir
    print '----------------------------------------------------'

    db_path = args.db_path
    raw_data_path = args.raw_data_path
    benchmark_dir = args.benchmark_dir
    num_data = args.num_data
    only_raw_data = args.only_raw_data
    is_shuffle = args.is_shuffle

    java = javac_parser.Java()

    initialize_database(db_path, raw_data_path, num_data, is_shuffle)
    if not only_raw_data:
        # initialize benchmark table
        print 'Start to process Benchmark'
        initialize_Benchmark_table(db_path, benchmark_dir)

