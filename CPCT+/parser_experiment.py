#encoding:  utf-8
import os
import javac_parser
import argparse
import time
import prettytable as pt
import numpy as np
from threading import Thread
from threading import Lock
from queue import Queue

def get_token_table():
    token_table = {}
    with open('./java7.l') as f:
        lines = f.readlines()
        for line in lines[1:]:
            if line == '\n':
                continue
            token_table[line.split()[1].replace('"', '')] = line.split()[0].replace('\\', '')

    # 这样提取并不准确 有些内容需要手动调整
    # 也许将来可以用正则表达式来 但是这些对语法错误并不重要
    # 或者想办法记录token-sequence顺序 然后再还原
    token_table['IDENTIFIER'] = 'mock_identifier'
    token_table['FLOATING_POINT_LITERAL'] = '0.1'
    token_table['CHARACTER_LITERAL'] = '\'c\''
    token_table['INTEGER_LITERAL'] = '0'
    token_table['BOOLEAN_LITERAL'] = 'true'
    token_table['STRING_LITERAL'] = '\"mock_string\"'

    return token_table


def is_successful_parse(file_path):
    '''
    给定文件的路径
    返回是否能够被成功parse
    '''
    use_parse_cmd = './java_parser_none ' + file_path

    f = os.popen(use_parse_cmd)
    output_content = f.read()
    if 'Parsing did not complete' in output_content:
        return False
    else:
        return True
    

def get_repair_output(file_path, repair_mode='CPCT_PLUS'):
    '''
    给定文件路径
    以文件的形式，返回得到的输出的结果
    这里修改下，能够自己选择使用哪个方法
    '''
    if repair_mode == 'CPCT_PLUS':
        use_parse_cmd = './nimbleparse ./java7.l ./java7.y ' + file_path
    if repair_mode == 'PANIC':
        use_parse_cmd = './nimbleparse -r panic ./java7.l ./java7.y ' + file_path

    f = os.popen(use_parse_cmd)

    return f

def get_tokens_from_output(f):
    parse_lines = f.readlines()
    tokens = []
    for line in parse_lines:
        if line == '\n':
            break
        useful_token = line.split()
        if useful_token[0] in token_table:
            if len(useful_token) == 1 or useful_token[0] == 'CHARACTER_LITERAL':
                # 即为 后面缺失的情况
                # 我们到表中进行查找
                tokens.append(token_table[line.split()[0]])
            else:
                concrete = ''.join(line.split()[1:])
                if '\\' in concrete:
                    tokens.append(token_table[line.split()[0]])
                else:
                    tokens.append(concrete)
    return tokens


def get_program(parse_output):
    '''
    这个函数用来处理nimbleparse的输出
    通过分析输出来得到原来的token序列
    '''
    return output


token_dict = {'~': '\n', u'EOF': u'', u'VOID': u'void', u'LBRACKET': u'[', u'SUBEQ': u'-=', u'PROTECTED': u'protected', u'AMPAMP': u'&&', u'TRUE': u'true', u'RPAREN': u')', u'LONG': u'long', u'PLUS': u'+', u'IMPORT': u'import', u'GTGTGT': u'>>>', u'GT': u'>', u'RBRACE': u'}', u'SUBSUB': u'--', u'ENUM': u'enum', u'EXTENDS': u'extends', u'SLASH': u'/', u'BAREQ': u'|=', u'THIS': u'this', u'DOUBLE': u'double', u'THROWS': u'throws', u'QUES': u'?', u'INTERFACE': u'interface', u'CARET': u'^', u'SHORT': u'short', u'STAR': u'*', u'SYNCHRONIZED': u'synchronized', u'EQEQ': u'==', u'STATIC': u'static', u'NEW': u'new', u'GTGT': u'>>', u'FINAL': u'final', u'SLASHEQ': u'/=', u'COLON': u':', u'AMP': u'&', u'FOR': u'for', u'PLUSPLUS': u'++', u'ELSE': u'else', u'TRY': u'try', u'STAREQ': u'*=', u'EQ': u'=', u'INSTANCEOF': u'instanceof', u'LBRACE': u'{', u'MONKEYS_AT': u'@', u'FINALLY': u'finally', u'CONTINUE': u'continue', u'SUB': u'-', u'GTEQ': u'>=', u'DEFAULT': u'default', u'CHAR': u'char', u'WHILE': u'while', u'RETURN': u'return', u'DOT': u'.', u'CASE': u'case', u'SWITCH': u'switch', u'CATCH': u'catch', u'PACKAGE': u'package', u'PERCENT': u'%', u'ELLIPSIS': u'...', u'BYTE': u'byte', u'IMPLEMENTS': u'implements', u'FALSE': u'false', u'LTEQ': u'<=', u'BREAK': u'break', u'INT': u'int', u'BOOLEAN': u'boolean', u'PUBLIC': u'public', u'DO': u'do', u'BAR': u'|', u'COLCOL': u'::', u'ABSTRACT': u'abstract', u'ASSERT': u'assert', u'LTLTEQ': u'<<=', u'BARBAR': u'||', u'NULL': u'null', u'SEMI': u';', u'PRIVATE': u'private', u'LT': u'<', u'COMMA': u',', u'CLASS': u'class', u'BANGEQ': u'!=', u'BANG': u'!', u'LPAREN': u'(', u'IF': u'if', u'PLUSEQ': u'+=', u'FLOAT': u'float', u'PERCENTEQ': u'%=', u'RBRACKET': u']', u'SUPER': u'super', u'THROW': u'throw'}

special_tokens_1 = '.()[]\{\};,'
special_tokens_2 = '.([\{'

line_num_tokens = '0123456789'
token_table = get_token_table()


def tokens_to_source(tokens):
    '''
    给定tokens的序列
    还原为程序
    '''
    source_program = ''
    new_tokens = tokens

    source_token = []
    token_length = len(new_tokens)
    for i, token in enumerate(new_tokens):
        if token in special_tokens_2:
            source_token.append(token)
            continue
        
        if i + 1 < token_length:
            if new_tokens[i+1] in special_tokens_1:
                source_token.append(token)
                continue

        source_token.append(token + ' ')
            
    return ''.join(source_token)

def repair_rate_analysis(result):
    java = javac_parser.Java()
    repair_rate_data = {'0~50' : [0, 0, 0, 0, 0], 
                        '50~100' : [0, 0, 0, 0, 0], 
                        '100~200': [0, 0, 0, 0, 0], 
                        '200~450': [0, 0, 0, 0, 0], 
                        '450~1000': [0, 0, 0, 0, 0]}
    
    original_total_error_num = 0
    repaired_total_error_num = 0

    original_err_msg_list = []
    repaired_err_msg_list = []

    too_large_count = 0
    zero_count = 0

    for token_num, org_error_list, repair_error_list in result:
        if token_num > 1000:
            too_large_count += 1
        if token_num == 0:
            zero_count += 1
        # 现在数量应该是对的了
        for key in repair_rate_data.keys():
            left = int(key.split('~')[0])
            right = int(key.split('~')[1])
            if token_num > left and token_num <= right:
                repair_rate_data[key][0] += 1

                # get the number of parse errors
                original_error_num = len(org_error_list)

                # store the eoors
                original_err_msg_list.append(org_error_list)

                # add errors to total
                original_total_error_num += original_error_num

                # do the same thing for repaired programs
                repaired_error_num = len(repair_error_list)
                repaired_err_msg_list.append(repair_error_list)
                repaired_total_error_num += repaired_error_num


                # Judge repair type: complete, partial, cascading or dumy (no change)
                is_complete_repair = False
                if original_error_num > 0 and repaired_error_num == 0:
                    # 如果修复后的为0，
                    is_complete_repair = True
                
                if is_complete_repair:
                    repair_rate_data[key][1] += 1

                is_partial_repair = False
                if original_error_num > repaired_error_num and repaired_error_num > 0:
                    is_partial_repair = True
                
                if is_partial_repair:
                    repair_rate_data[key][2] += 1

                introduce_cascading_error = False
                if original_error_num < repaired_error_num:
                    introduce_cascading_error = True
                
                if introduce_cascading_error:
                    repair_rate_data[key][3] += 1

                
                dumy_repair = False
                if original_error_num == repaired_error_num:
                    ## 如果前后相同，且不为0
                    dumy_repair = True
                if dumy_repair:
                    repair_rate_data[key][4] += 1


        # store data by lines of code
    print('> 1000: ', too_large_count)
    print('Zero Count: ', zero_count)

    tb = pt.PrettyTable()
    tb.field_names = ['Token Length Range', 'Number of Files', 'Complete Repair', 
                        'Partial Repair', 'Cascading Repair', 'Dummy Repair']
    for key in repair_rate_data.keys():
        tb.add_row([key] + repair_rate_data[key])

    print(tb)


def time_analysis(time_consumed):
    time_data = {}
    tb = pt.PrettyTable()
    tb.field_names = ['Token Length Range', 'Average Repair Time']
    for row in time_consumed:
        token_length = row[0]
        time = row[1]

        for left in range(100):
            right = left + 1
            if token_length > left * 10 and token_length <= right * 10:
                key  = str(left*10) + '~' + str(right*10)
                try:
                    time_data[key].append(time)
                except Exception:
                    time_data[key] = [time]
    
    for left in range(100):
        right = left + 1
        key  = str(left*10) + '~' + str(right*10)
        try:
            average_time = np.sum(time_data[key]) / (len(time_data[key]) + 0.001) * 1000
        except:
            continue
        tb.add_row([key, average_time])
    
    print(tb)
    return 

def count_err_message(err_msg_list):
    '''
    input: error messages list
    output: distribution of types of messages
    '''
    err_distribution = {}
    for err_list in err_msg_list:
        for error in err_list:
            if error[0] == 'ERROR':
                if error[1] in err_distribution:
                    err_distribution[error[1]] += 1
                else:
                    err_distribution[error[1]] = 1

    return err_distribution

def analyze_perf_by_err_type(result):
    '''
    Analyze the repair performance on different types of errors
    Input: two dictionary, 1: err type distribution before; 2: .. after repair
    Output: A dictionary, {erro_type: [before_num, after_num], ..., ...}
    '''
    original_err_msg_list = []
    repaired_err_msg_list = []
    for token_num, org_error_list, repair_error_list in result:
        original_err_msg_list.append(org_error_list)
        repaired_err_msg_list.append(repair_error_list)

    original_err_distribution = count_err_message(original_err_msg_list)
    repaired_err_distribution = count_err_message(repaired_err_msg_list)

    error_dict = {}
    for key in original_err_distribution.keys():
        if key in error_dict:
            error_dict[key][0] = original_err_distribution[key]
        else:
            error_dict[key] = [0,0,0]
            error_dict[key][0] = original_err_distribution[key]
    
    for key in repaired_err_distribution:
        if key in error_dict:
            error_dict[key][1] = repaired_err_distribution[key]
        else:
            error_dict[key] = [0,0,0]
            error_dict[key][1] = repaired_err_distribution[key]  

    error_cnt_before = 0
    error_cnt_after = 0

    for key in error_dict:
        error_cnt_before += error_dict[key][0]
        error_cnt_after += error_dict[key][1]
        if error_dict[key][0] == 0:
            error_dict[key][2] = 'NAN'
        else:
            resolve_rate = (error_dict[key][0] - error_dict[key][1]) * 100.0 / error_dict[key][0]
            error_dict[key][2] = str(round(resolve_rate, 2)) + '%'

    tb = pt.PrettyTable()
    tb.field_names = ['Error Type', 'Before', 'After', 'Resolve Rate']
    for key in error_dict.keys():
        tb.add_row([key] + error_dict[key])
    total_resovle_rate = (error_cnt_before - error_cnt_after) * 100.0 / (error_cnt_before + 0.001)
    tb.add_row(['Total', error_cnt_before, error_cnt_after, str(round(total_resovle_rate, 2)) + '%'])
    # tb.add_row(['Total', len(dataset)])

    print(tb)

def post_analysis(result, time_consumed):
    # 分析4个repair rate
    repair_rate_analysis(result)

    # 分析时间消耗
    time_analysis(time_consumed)

    # 分析在不同类型错误上的表现
    analyze_perf_by_err_type(result)

class AtomicInt:
    def __init__(self, v):
        self.v = v
        self._lock = Lock()

    def add(self, i):
        with self._lock:
            o = self.v
            self.v = o + i
        return o

    def val(self):
        with self._lock:
            return self.v


class Worker(Thread):
    def run(self):
        while True:
            i = generated.add(1)
            if i >= number:
                exit()
            java_file_path = q.get()
            with open(java_file_path) as f:
                org_code = f.read()
                # 得到原来的error list
                try:
                    org_error_list = java.check_syntax(org_code)
                except:
                    q.task_done()
                    continue
                    
            start = time.time()
            # 得到经过修复后的内容，存在一个文件中
            f = get_repair_output(java_file_path, repair_mode=mode)
            # 处理此文件，提取出tokens
            tokens = get_tokens_from_output(f)
            # 同时得到tokens的长度
            token_length = len(tokens)
            # 将这些tokens还原为源文件
            source_program = tokens_to_source(tokens)
            time_consumed = time.time() - start


            # 得到修复后的错误列表
            try:
                repair_error_list = java.check_syntax(source_program)
            except:
                q.task_done()
                continue
            values_to_log = [token_length, org_error_list, repair_error_list]
            result.append(values_to_log)
            time_list.append([token_length,time_consumed])

            q.task_done()


parser = argparse.ArgumentParser(description="分析CPCT+算法的修复结果")
parser.add_argument("-n", "--number", help="分析的文件数量", type=int, default=200000)
parser.add_argument("source_file_dir", help="src_files这个文件夹的路径")
parser.add_argument("-m", "--mode", help="recovery模式(CPCT_PLUS or PANIC)", default='CPCT_PLUS')

args = parser.parse_args()

java = javac_parser.Java()
count = 0

result = []
time_list = []

source_file_dir = args.source_file_dir
number = args.number
mode = args.mode
print('--------------------------')
print('%s mode selected' % (mode))
print('--------------------------')


q = Queue(200000)

generated = AtomicInt(0)

workers = []

for java_file_name in os.listdir(source_file_dir):
    
    java_file_path = os.path.join(source_file_dir, java_file_name)
    q.put(java_file_path)


for _ in range(10):
    w = Worker()
    w.daemon = True
    workers.append(w)
    w.start()



for w in workers:
    w.join()

print('Total programs processed: ', len(result))
post_analysis(result, time_list)

        
    
    