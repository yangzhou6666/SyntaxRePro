#encoding:  utf-8
import os
import sqlite3
import json
import argparse
import time
import math
import numpy as np
import tensorflow as tf
import javac_parser
# from data_processing.training_data_generator import vectorize
from generate_java_data import vectorize
from neural_net.train import load_data, seq2seq_model as model
from post_processing.postprocessing_helpers import devectorize, \
    VectorizationFailedException
# from util.helpers import apply_fix, vstack_with_right_padding, make_dir_if_not_exists
from util.helpers import apply_fix_for_java, vstack_with_right_padding, make_dir_if_not_exists

from util.helpers import InvalidFixLocationException, SubstitutionFailedException
import prettytable as pt




parser = argparse.ArgumentParser(
    description="Predict and store fixes in a database.")

parser.add_argument("checkpoint_directory", help="Checkpoint directory")
parser.add_argument('-dd', "--data_directory", help="Data directory")
# parser.add_argument('-d', "--database", help="sqlite3 database to use")
parser.add_argument('-w', '--which', help='Which test data.',
                    choices=['raw', 'seeded'], default="raw")
parser.add_argument('-b', '--batch_size', type=int,
                    help="batch_size", default=128)
parser.add_argument("--embedding_dim", type=int,
                    help="embedding_dim", default=50)
parser.add_argument('-m', "--memory_dim", type=int,
                    help="memory_dim", default=300)
parser.add_argument('-n', "--num_layers", type=int,
                    help="num_layers", default=4)
parser.add_argument('-c', '--cell_type',
                    help='One of RNN, LSTM or GRU.', default="GRU")
parser.add_argument(
    '-v', '--vram', help='Fraction of GPU memory to use', type=float, default=0.7)
parser.add_argument('-a', '--max_attempts',
                    help='How many iterations to limit the repair to', type=int, default=1)
parser.add_argument("-r", "--resume_at", type=int,
                    help="Checkpoint to resume from (leave blank to resume from the best one)", default=None)
parser.add_argument("-t", "--task", help="Specify the task for which the network has been trained",
                    choices=['typo', 'ids'], default='typo')
parser.add_argument("--max_prog_length", type=int,
                    help="maximum length of the programs in tokens", default=450)
parser.add_argument('-o', '--max_output_seq_len',
                    help='max_output_seq_len', type=int, default=28)
parser.add_argument('--is_timing_experiment', action="store_true",
                    help="This is a timing experiment, do not store results")
parser.add_argument('-ep', '--evaluation_path', default='data/network_inputs/Deepfix-Java-seed-1189/bin_1/test_raw_bin_1.npy',
                    help='Path to evaluation.npy')
parser.add_argument('-en','--evaluate_number', default='10', help='Number of programs to evaluate')

args = parser.parse_args()

evaluate_number = args.evaluate_number

# database_path = args.checkpoint_directory.replace(
#     '''data/checkpoints/''', '''data/results/''')
args.checkpoint_directory = args.checkpoint_directory + \
    ('' if args.checkpoint_directory.endswith('/') else '/')
bin_id = None
try:
    if args.checkpoint_directory.find('bin_') == -1:
        raise ValueError('ERROR: failed to find the bin id')
    bin_id = int(args.checkpoint_directory[-2])
    print 'bin_id:', bin_id
except:
    raise

# if args.database:
#     database = args.database
# else:
#     make_dir_if_not_exists(database_path)
#     database_name = args.which + '_' + args.task + '.db'
#     database = os.path.join(database_path, database_name)

# print 'using database:', database

if not args.data_directory:
    # 如果没有指定data_directory
    # 则从checkpoint中读取configuration文件，来得到文件路径
    training_args = np.load(os.path.join(
        args.checkpoint_directory, 'experiment-configuration.npy'), allow_pickle=True).item()['args']

    args.data_directory = training_args.data_directory

print 'data directory:', args.data_directory


# Checkpoint information
if args.resume_at is None:
    best_checkpoint = None

    for checkpoint_name in os.listdir(os.path.join(args.checkpoint_directory, 'best')):
        if 'meta' in checkpoint_name:
            this_checkpoint = int(checkpoint_name[17:].split('.')[0])

            if best_checkpoint is None or this_checkpoint > best_checkpoint:
                best_checkpoint = this_checkpoint

    print "Resuming at", best_checkpoint, "..."
else:
    best_checkpoint = args.resume_at

# Load data
dataset = load_data(args.data_directory, shuffle=False, load_only_dicts=True)
dictionary = dataset.get_tl_dictionary()

# Build the network
# scope = 'typo' if 'typo' in args.data_directory else 'ids'
scope = 'typo'

with tf.variable_scope(scope):
    seq2seq = model(dataset.vocabulary_size, args.embedding_dim,
                    args.max_output_seq_len,
                    cell_type=args.cell_type,
                    memory_dim=args.memory_dim,
                    num_layers=args.num_layers,
                    dropout=0,
                    )


def get_fix(sess, program):
    X, X_len = tuple(dataset.prepare_batch(program))
    return seq2seq.sample(sess, X, X_len)[0]


def get_fixes_in_batch(sess, programs):
    X, X_len = tuple(dataset.prepare_batch(programs))
    fixes = seq2seq.sample(sess, X, X_len)
    assert len(programs) == np.shape(fixes)[0]
    return fixes


def get_fixes(sess, programs):
    num_programs = len(programs)
    all_fixes = []

    for i in range(int(math.ceil(num_programs * 1.0 / args.batch_size))):
        start = i * args.batch_size
        end = (i + 1) * args.batch_size
        fixes = get_fixes_in_batch(sess, programs[start:end])
        all_fixes.append(fixes)

    fixes = vstack_with_right_padding(all_fixes)
    assert num_programs == np.shape(fixes)[
        0], 'num_programs: {}, fixes-shape: {}'.format(num_programs, np.shape(fixes))

    return fixes

def count_token_length(dataset):
    '''
    输入：存储了一组程序信息的dictionary
    输出：程序的长度分布
    '''
    token_length_distribution = np.zeros(50000)

    for problem_id, test_programs in dataset.iteritems():
        token_seq = test_programs[0][0]
        token_len = len(token_seq.split())
        token_length_distribution[token_len] += 1
    
    ranges = {'0~50' : 0, '50~100' : 0, '100~200': 0, '200~450': 0, '450~1000': 0}
    other = 0
    

    for i, token_len_data in enumerate(token_length_distribution):
        for key in ranges.keys():
            left = int(key.split('~')[0])
            right = int(key.split('~')[1])
            if i >= left and i <= right:
                ranges[key] += token_len_data
        

    tb = pt.PrettyTable()
    tb.field_names = ['Token Length Range', 'Number of Files']
    for key in ranges.keys():
        tb.add_row([key, ranges[key]])
    tb.add_row(['Other', other])
    tb.add_row(['Total', len(dataset)])

    print tb

    return ranges

def count_LoC(dataset):
    '''
    input :  evaluation program batch
    output:  a list that stores lines of code
    '''
    loc_distribution = np.zeros(50000)
    for problem_id, test_programs in dataset.iteritems():
        for program, name_dict, name_sequence, user_id, program_id in [test_programs]:
            line_num = 0
            for token in program.split():
                if token == '~':
                    line_num += 1
            loc_distribution[line_num] += 1

    ranges = np.zeros(5)
    # 为了方便使用prettytable，这里用字典可能更方便一点

    for i, loc_data in enumerate(loc_distribution):
        if i >= 4 and i <= 15:
            ranges[0] += loc_data
        if i > 15 and i <= 50:
            ranges[1] += loc_data
        if i > 50 and i <= 100:
            ranges[2] += loc_data
        if i > 100 and i <= 200:
            ranges[3] += loc_data
        if i > 200:
            ranges[4] += loc_data

    tb = pt.PrettyTable()
    tb.field_names = ['Token Length Range', 'Number of Files']

    tb.add_row(['4~15'], ranges[0])
    tb.add_row(['15~50'], ranges[1])
    tb.add_row(['50~100'], ranges[2])
    tb.add_row(['100~200'], ranges[3])
    tb.add_row(['> 200'], ranges[4])

    print tb

    return ranges

token_dict = {'~': '\n', u'EOF': u'', u'VOID': u'void', u'LBRACKET': u'[', u'SUBEQ': u'-=', u'PROTECTED': u'protected', u'AMPAMP': u'&&', u'TRUE': u'true', u'RPAREN': u')', u'LONG': u'long', u'PLUS': u'+', u'IMPORT': u'import', u'GTGTGT': u'>>>', u'GT': u'>', u'RBRACE': u'}', u'SUBSUB': u'--', u'ENUM': u'enum', u'EXTENDS': u'extends', u'SLASH': u'/', u'BAREQ': u'|=', u'THIS': u'this', u'DOUBLE': u'double', u'THROWS': u'throws', u'QUES': u'?', u'INTERFACE': u'interface', u'CARET': u'^', u'SHORT': u'short', u'STAR': u'*', u'SYNCHRONIZED': u'synchronized', u'EQEQ': u'==', u'STATIC': u'static', u'NEW': u'new', u'GTGT': u'>>', u'FINAL': u'final', u'SLASHEQ': u'/=', u'COLON': u':', u'AMP': u'&', u'FOR': u'for', u'PLUSPLUS': u'++', u'ELSE': u'else', u'TRY': u'try', u'STAREQ': u'*=', u'EQ': u'=', u'INSTANCEOF': u'instanceof', u'LBRACE': u'{', u'MONKEYS_AT': u'@', u'FINALLY': u'finally', u'CONTINUE': u'continue', u'SUB': u'-', u'GTEQ': u'>=', u'DEFAULT': u'default', u'CHAR': u'char', u'WHILE': u'while', u'RETURN': u'return', u'DOT': u'.', u'CASE': u'case', u'SWITCH': u'switch', u'CATCH': u'catch', u'PACKAGE': u'package', u'PERCENT': u'%', u'ELLIPSIS': u'...', u'BYTE': u'byte', u'IMPLEMENTS': u'implements', u'FALSE': u'false', u'LTEQ': u'<=', u'BREAK': u'break', u'INT': u'int', u'BOOLEAN': u'boolean', u'PUBLIC': u'public', u'DO': u'do', u'BAR': u'|', u'COLCOL': u'::', u'ABSTRACT': u'abstract', u'ASSERT': u'assert', u'LTLTEQ': u'<<=', u'BARBAR': u'||', u'NULL': u'null', u'SEMI': u';', u'PRIVATE': u'private', u'LT': u'<', u'COMMA': u',', u'CLASS': u'class', u'BANGEQ': u'!=', u'BANG': u'!', u'LPAREN': u'(', u'IF': u'if', u'PLUSEQ': u'+=', u'FLOAT': u'float', u'PERCENTEQ': u'%=', u'RBRACKET': u']', u'SUPER': u'super', u'THROW': u'throw'}

special_tokens_1 = '.()[]\{\};,'
special_tokens_2 = '.([\{'
line_num_tokens = '0123456789'


def get_tokenized_loc(tokenized_code):
    line_num = 0
    token_num = len(tokenized_code.split())
    
    for token in tokenized_code.split():
        if token == '~':
            line_num += 1 
    return token_num, line_num

def tokens_to_source(program, name_dict):
    source_program = ''
    new_dict = {v : k for k, v in name_dict.items()}
    tokens = program.split()
    new_tokens = []
    for token in tokens:
        if 'IDENTIFIER' in token:
            # 奇怪的表示 需要重构
            identifier_info = token[10:].split('@')
            # print token
            if len(identifier_info) > 1:
                new_tokens.append(new_dict[int(identifier_info[0])])
            else:
                new_tokens.append('new_id')
        else:
            if token in token_dict.keys():
                new_tokens.append(token_dict[token])
            else:
                new_tokens.append(token)

    source_token = []
    token_length = len(new_tokens)
    for i, token in enumerate(new_tokens):
        if token in line_num_tokens:
            continue
        if token in special_tokens_2:
            source_token.append(token)
            continue
        
        if i + 1 < token_length:
            if new_tokens[i+1] in special_tokens_1:
                source_token.append(token)
                continue

        source_token.append(token + ' ')
            

    return ''.join(source_token)


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

def time_analysis(times, token_lens):

    time_data = {}

    for i in range(len(times)):
        for left in range(100):
            right = left + 1
            if token_lens[i] >= left * 10 and token_lens[i] <= right * 10:
                key = str( left * 10)+'~'+str(right * 10)
                try:
                    time_data[key].append(times[i])
                except Exception:
                    time_data[key] = [times[i]]   

    tb = pt.PrettyTable()
    tb.field_names = ['Token Length Range', 'Average Repair Time']
    for left in range(100):
        right = left + 1
        key  = str(left*10) + '~' + str(right*10)
        try:
            average_time = np.sum(time_data[key]) / (len(time_data[key]) + 0.001) * 1000
        except:
            continue
        tb.add_row([key, average_time])

    print tb

    return

def analyze_perf_by_err_type(original_err_distribution, repaired_err_distribution):
    '''
    Analyze the repair performance on different types of errors
    Input: two dictionary, 1: err type distribution before; 2: .. after repair
    Output: A dictionary, {erro_type: [before_num, after_num], ..., ...}
    '''

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
            error_dict[key][2] = str(round((error_dict[key][0] - error_dict[key][1]) * 100.0 / error_dict[key][0])) + '%'

    tb = pt.PrettyTable()
    tb.field_names = ['Error Type', 'Before', 'After', 'Resolve Rate']
    for key in error_dict.keys():
        tb.add_row([key] + error_dict[key])
    tb.add_row(['Total', error_cnt_before, error_cnt_after, 
                str(round((error_cnt_before - error_cnt_after) * 100.0 / error_cnt_before)) + '%'])
    # tb.add_row(['Total', len(dataset)])

    print tb


def post_analysis(before_after_program):
    java = javac_parser.Java()
    repair_rate_data = {'0~50' : [0, 0, 0, 0, 0], 
                        '50~100' : [0, 0, 0, 0, 0], 
                        '100~200': [0, 0, 0, 0, 0], 
                        '200~450': [0, 0, 0, 0, 0], 
                        '450~1000': [0, 0, 0, 0, 0]}

    complete_rate_data = {'< 4': [0, 0, 0, 0, 0], \
                            '4~15': [0, 0, 0, 0, 0], \
                            '15~50': [0, 0, 0, 0, 0], \
                            '50~100': [0, 0, 0, 0, 0], \
                            '100~200': [0, 0, 0, 0, 0], \
                            '> 200': [0, 0, 0, 0, 0]}

    original_total_error_num = 0
    repaired_total_error_num = 0

    original_err_msg_list = []
    repaired_err_msg_list = []



    for original_program, repaired_program, token_num, line_num in before_after_program:
        for key in repair_rate_data.keys():
            left = int(key.split('~')[0])
            right = int(key.split('~')[1])
            if token_num >= left and token_num <= right:
                repair_rate_data[key][0] += 1

                # use javac_parser.check_syntax() to get all the parse errors
                original_err = java.check_syntax(original_program)

                # get the number of parse errors
                original_error_num = len(original_err)

                # store the eoors
                original_err_msg_list.append(original_err)

                # add errors to total
                original_total_error_num += original_error_num

                # do the same thing for repaired programs
                after_err = java.check_syntax(repaired_program)
                repaired_error_num = len(after_err)
                repaired_err_msg_list.append(after_err)
                repaired_total_error_num += repaired_error_num


                # Judge repair type: complete, partial, cascading or dumy (no change)
                is_complete_repair = False
                if repaired_error_num == 0:
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
                    dumy_repair = True
                if dumy_repair:
                    repair_rate_data[key][4] += 1


        # store data by lines of code

    tb = pt.PrettyTable()
    tb.field_names = ['Token Length Range', 'Number of Files', 'Complete Repair', 
                        'Partial Repair', 'Cascading Repair', 'Dummy Repair']
    for key in repair_rate_data.keys():
        tb.add_row([key] + repair_rate_data[key])
    # tb.add_row(['Other', other])
    # tb.add_row(['Total', len(dataset)])

    print tb
    
    # print complete_rate_data

    # print 'original_total_error_num:  %d' % (original_total_error_num)
    # print 'repaired_total_error_num  :  %d ' % (repaired_total_error_num)
    original_err_distribution = count_err_message(original_err_msg_list)
    repaired_err_distribution = count_err_message(repaired_err_msg_list)
    analyze_perf_by_err_type(original_err_distribution, repaired_err_distribution)


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.vram)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

if args.resume_at is None:
    seq2seq.load_parameters(sess, os.path.join(
        args.checkpoint_directory, 'best', 'saved-model-attn-' + str(best_checkpoint)))
else:
    seq2seq.load_parameters(sess, os.path.join(
        args.checkpoint_directory, 'saved-model-attn-' + str(best_checkpoint)))

# 这里读取了evaluation_path下的文件
test_dataset = np.load(args.evaluation_path, allow_pickle=True).item()
token_length_data = count_token_length(test_dataset)



# Attempt to repair
sequences_of_programs = {}
fixes_suggested_by_network = {}


if args.task == 'typo':
    normalize_names = True
    fix_kind = 'replace'
else:
    assert args.task == 'ids'
    normalize_names = False
    fix_kind = 'insert'

times = []
counts = []
token_lens = []

print 'test data length:', len(test_dataset)

java = javac_parser.Java()



before_after_program = []

count = 0

for problem_id, test_programs in test_dataset.iteritems():
    if count % 10000 == 0:
        print '%d programs evaluated' % (count)
    count += 1
    sequences_of_programs[problem_id] = {}
    fixes_suggested_by_network[problem_id] = {}
    start = time.time()

    token_seq = test_programs[0][0]
    token_len = len(token_seq.split())

    entries = []
    for program, name_dict, name_sequence, program_id in test_programs:

        sequences_of_programs[problem_id][program_id] = [program]
        fixes_suggested_by_network[problem_id][program_id] = []
        entries.append(
            (program, name_dict, name_sequence, program_id,))

    for round_ in range(args.max_attempts):

        to_delete = []
        input_ = []
        for i, entry in enumerate(entries):
            program, name_dict, name_sequence, program_id = entry

            try:
                program_vector = vectorize(sequences_of_programs[problem_id][program_id][-1], dictionary, args.max_prog_length,
                                           normalize_names, reverse=True)
            except:
                program_vector = None

            if program_vector is not None:
                input_.append(program_vector)
            else:
                to_delete.append(i)

        to_delete = sorted(to_delete)[::-1]

        for i in to_delete:
            del entries[i]

        assert len(input_) == len(entries)

        if len(input_) == 0:
            # print 'Stopping before iteration %d (no programs remain)' % (round_ + 1)
            break

        # Get fix predictions
        fix_vectors = get_fixes(sess, input_)
        fixes = []

        # devectorize fixes
        for i, fix_vector in enumerate(fix_vectors):
            _, _, _, program_id = entries[i]
            fix = devectorize(fix_vector, dictionary)
            fixes_suggested_by_network[problem_id][program_id].append(fix)
            fixes.append(fix)

        to_delete = []

        # Apply fixes
        for i, entry, fix in zip(range(len(fixes)), entries, fixes):
            program, name_dict, name_sequence, program_id = entry
            try:
                program = sequences_of_programs[problem_id][program_id][-1]
                # get token_num and LoC
                token_num, line_num = get_tokenized_loc(program)

                original_program = tokens_to_source(program, name_dict)

                program = apply_fix_for_java(program, fix, fix_kind,
                                    flag_replace_ids=False)
                repaired_program = tokens_to_source(program, name_dict)

                # store (original program, repaired program, token number, loc)
                before_after_program.append((original_program, repaired_program, token_num, line_num))

                sequences_of_programs[problem_id][program_id].append(program)
            except ValueError as e:
                to_delete.append(i)
                sequences_of_programs[problem_id][program_id].append(
                    '{{localization_failed}}')
            except VectorizationFailedException as e:
                to_delete.append(i)
                sequences_of_programs[problem_id][program_id].append(
                    '{{vectorization_failed %s}}' % e.args[0])
            except InvalidFixLocationException as e:
                to_delete.append(i)
                sequences_of_programs[problem_id][program_id].append(
                    '{{localization_failed}}')
            except SubstitutionFailedException as e:
                to_delete.append(i)
                sequences_of_programs[problem_id][program_id].append(
                    '{{back_substitution_failed}}')
            except Exception as e:
                raise e
        to_delete = sorted(to_delete)[::-1]

        for i in to_delete:
            del entries[i]

    times.append(time.time() - start)
    token_lens.append(token_len)
    counts.append(len(test_programs))
    if len(counts) >= 200000:
        break

assert len(times) == len(token_lens)

post_analysis(before_after_program)
time_analysis(times, token_lens)


print 'Total time:', np.sum(times), 'seconds'
print 'Total programs processed:', np.sum(counts)
print 'Average time per program:', int(float(np.sum(times)) / float(np.sum(counts)) * 1000), 'ms'