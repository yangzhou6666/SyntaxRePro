#encoding:  utf-8
import os
import sqlite3
import json
import sys
import javac_parser
import argparse

reload(sys)
sys.setdefaultencoding('utf8')

def initialize_database(db_path, raw_data_path, data_length):
    '''
    这里直接就生成tokenized code等信息并写入数据库，而不用二次写入
    '''
    # 初始化表的结构
    with sqlite3.connect(db_path) as conn:
        conn.execute('''CREATE TABLE Code (code_id, code text);''')
    
    # 增加一些新的columns
    with sqlite3.connect(db_path) as conn:
        conn.execute('''ALTER TABLE Code ADD tokenized_code text;''')
        conn.execute('''ALTER TABLE Code ADD name_dict;''')
        conn.execute('''ALTER TABLE Code ADD name_seq;''')
        conn.execute('''ALTER TABLE Code ADD codelength integer;''')
    
    # count是已经生成数据数量
    count = 0
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        for file_event_id in os.listdir(raw_data_path):
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
    接受数据库和原数据的路径，将正确的文件写入数据库
    '''
    # 现在开始处理benchmark
    with sqlite3.connect(db_path) as conn:
        conn.execute('''DROP TABLE Benchmark''')
        conn.execute('''CREATE TABLE Benchmark (code_id, code text);''')

    with sqlite3.connect(db_path) as conn:
        conn.execute('''ALTER TABLE Benchmark ADD tokenized_code text;''')
        conn.execute('''ALTER TABLE Benchmark ADD name_dict;''')
        conn.execute('''ALTER TABLE Benchmark ADD name_seq;''')
        conn.execute('''ALTER TABLE Benchmark ADD codelength integer;''')
    
    # count是已经生成数据数量
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

def update_tokenized_Code(db_path):
    '''
    接受数据库路径，修改数据表，并使用tokenized code来修改数据
    '''
    # 修改数据表结构
    with sqlite3.connect(db_path) as conn:
        conn.execute('''ALTER TABLE Code ADD tokenized_code text;''')
        conn.execute('''ALTER TABLE Code ADD name_dict;''')
        conn.execute('''ALTER TABLE Code ADD name_seq;''')
        conn.execute('''ALTER TABLE Code ADD codelength integer;''')

    tuples = []
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        for row in cursor.execute("SELECT code_id, code FROM Code;"):
            code_id = str(row[0])
            code = row[1].encode('utf-8')
            tokenized_code, name_dict, name_seq = get_tokenized_code(code)
            # 这里的code_length并不是原始code的token数量，而是tokenized过后的
            codelength = len(tokenized_code.split())
            tuples.append((tokenized_code, json.dumps(name_dict),
                        json.dumps(name_seq), codelength, code_id))

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        print 'Start to update database'
        cursor.executemany(
            "UPDATE Code SET tokenized_code=?, name_dict=?, name_seq=?, codelength=? WHERE code_id=?;", tuples)
        conn.commit()
    
    
def get_tokenized_code(code):

    # code = "\n".join([s for s in code.splitlines() if s.strip()])

    tokens = []
    name_dict = {}
    name_seq = []
    loc = len(code.split('\n'))
    # tokens_by_line = [list(str(i)) for i in range(loc)]
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

    # 去除空白行
    # 判断空白行的方法是，看那一行的最后一个token是不是'~'
    tokens_by_line_no_blank_lines = []
    for line in tokens_by_line:
        if line[-1] == '~':
            continue
        tokens_by_line_no_blank_lines.append(line)
    # 更新行号
    for i, line in enumerate(tokens_by_line_no_blank_lines):
        line_num = i
        line[0] = ' '.join(list(str(line_num)))


    token_string_by_line = []
    for line in tokens_by_line_no_blank_lines:
        tokens.append(' '.join(line))
        token_string_by_line.append(' '.join(line))

    return ' '.join(token_string_by_line), name_dict, name_seq


def delete_empty_line(code):
    '''
    要求能去空行 + 去除只有注释的地方
    不然的话直接用javac_parser来lex，会产生非常多的空行
    另一个做法是，先lex，然后再去除空行，并修改一个个行号的位置
    '''
    lines = code.split('\n')
    new_code = '\n'.join(lines)
    return new_code

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Preprocess Java programs')
    parser.add_argument('-d', '--db_path', help='database path to store', 
                        default = 'data/java_data/java_data.db')
    parser.add_argument('-r', '--raw_data_path', help='raw java data path to preprocess',
                        default = '../../java_data/')
    parser.add_argument('-n', '--num_data', help='Number to data example to generate',
                        default = 10000)
    parser.add_argument('-bd', '--benchmark_dir', help = 'Path to src_files directory',
                        default = '../../src_files/')

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

    java = javac_parser.Java()

    initialize_database(db_path, raw_data_path, num_data)
    initialize_Benchmark_table(db_path, benchmark_dir)

