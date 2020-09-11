#encoding:  utf-8
import os
from util.helpers import get_rev_dict, make_dir_if_not_exists as mkdir, remove_empty_new_lines
import os, time, argparse, sqlite3, json, numpy as np
from util.c_tokenizer import C_Tokenizer
from functools import partial
from data_processing.typo_mutator import LoopCountThresholdExceededException, FailedToMutateException, Typo_Mutate, typo_mutate, Typo_Mutate_Java

def save_dictionaries(destination, tldict):
    all_dicts = (tldict, get_rev_dict(tldict))
    np.save(os.path.join(destination, 'all_dicts.npy'), all_dicts)

def save_pairs(destination, token_vectors, tldict, name_dict_store):
    np.save(os.path.join(destination, ('name_dict_store.npy')), name_dict_store )
    save_dictionaries(destination, tldict)
    for key in token_vectors.keys():
        np.save(os.path.join(destination, ('examples-%s.npy' % key)), token_vectors[key] )


def get_bins(db_path, min_program_length, max_program_length):
    bins = []
    with sqlite3.connect(db_path) as conn:
        code_id_list = []

        cursor = conn.cursor()
        query = "SELECT code_id FROM Code WHERE codelength>? and codelength<?;"
        for row in cursor.execute(query, (min_program_length, max_program_length)):
            # 得到所有的code_id
            code_id_list.append(row[0])
        
        length = len(code_id_list)  # 总长
        n = 5  # 切分成多少份
        step = int(length / n) + 1  # 每份的长度
        for i in range(0, length, step):
            bins.append(code_id_list[i: i + step])

    return bins


def save_bins(destination, tldict, token_vectors, bins, name_dict_store):
    full_list = []
    # 提取所有的problem_id放到一个list当中
    for bin_ in bins:
        for problem_id in bin_:
            full_list.append(problem_id)

    for i, bin_ in enumerate(bins):
        # 把当前的这个bin_当作test_problems
        test_problems = bin_
        # 通过差集得到剩下的，作为trainging_problems
        training_problems = list(set(full_list) - set(bin_))

        token_vectors_this_fold = { 'train': {}, 'validation': {}, 'test': {} }

        for problem_id in training_problems:
            if problem_id in token_vectors['train']:
                for code_id, inc_prog_vector, corr_prog_vector in token_vectors['train'][problem_id]:
                    variant = 1
                    temp_code_id = code_id
                    while temp_code_id in token_vectors_this_fold['train']:
                        temp_code_id = code_id + '_v%d' % variant
                        variant += 1
                    variant = 1
                    code_id = temp_code_id

                    token_vectors_this_fold['train'][code_id] = (inc_prog_vector, corr_prog_vector)

            if problem_id in token_vectors['validation']:
                for code_id, inc_prog_vector, corr_prog_vector in token_vectors['validation'][problem_id]:
                    variant = 1
                    temp_code_id = code_id
                    while temp_code_id in token_vectors_this_fold['validation']:
                        temp_code_id = code_id + '_v%d' % variant
                        variant += 1
                    variant = 1
                    code_id = temp_code_id

                    token_vectors_this_fold['validation'][code_id] = (inc_prog_vector, corr_prog_vector)

        for problem_id in test_problems:
            if problem_id in token_vectors['validation']:
                for code_id, inc_prog_vector, corr_prog_vector in token_vectors['validation'][problem_id]:
                    variant = 1
                    temp_code_id = code_id
                    while temp_code_id in token_vectors_this_fold['test']:
                        temp_code_id = code_id + '_v%d' % variant
                        variant += 1
                    variant = 1
                    code_id = temp_code_id

                    token_vectors_this_fold['test'][code_id] = (inc_prog_vector, corr_prog_vector)
            
            if problem_id in token_vectors['train']:
                for code_id, inc_prog_vector, corr_prog_vector in token_vectors['train'][problem_id]:
                    variant = 1
                    temp_code_id = code_id
                    while temp_code_id in token_vectors_this_fold['test']:
                        temp_code_id = code_id + '_v%d' % variant
                        variant += 1
                    variant = 1
                    code_id = temp_code_id

                    token_vectors_this_fold['test'][code_id] = (inc_prog_vector, corr_prog_vector)
        mkdir(os.path.join(destination, 'bin_%d' % i))

        print "Fold %d: %d Train %d Validation %d Test" % (i, len(token_vectors_this_fold['train']), \
                                                            len(token_vectors_this_fold['validation']), \
                                                            len(token_vectors_this_fold['test']))
        save_pairs(os.path.join(destination, 'bin_%d' % i), token_vectors_this_fold, tldict, name_dict_store)


def vectorize(tokens, tldict, max_vector_length):
    vec_tokens = []
    for token in tokens.split():
        try:
            vec_tokens.append(tldict[token])
        except Exception:
            print token
            raise

    if len(vec_tokens) > max_vector_length:
        return None

    return vec_tokens

def vectorize_data(token_strings, tldict, max_program_length):
    token_vectors = {}
    skipped = 0

    for key in token_strings:
        token_vectors[key] = {}
        for problem_id in token_strings[key]:
            token_vectors[key][problem_id] = []

    for key in token_strings:
        for problem_id in token_strings[key]:
            for code_id, prog_tokens, fix_tokens in token_strings[key][problem_id]:
                inc_prog_vector = vectorize(prog_tokens, tldict, max_program_length)
                corr_prog_vector = vectorize(fix_tokens, tldict,  max_program_length)

                if (inc_prog_vector is not None) and (corr_prog_vector is not None):
                    token_vectors[key][problem_id].append((code_id, inc_prog_vector, corr_prog_vector))
                else:
                    skipped += 1

    print 'skipped during vectorization:', skipped
    return token_vectors

def build_dictionary(token_strings, tldict={}):

    def build_dict(list_generator, dict_ref):
        for tokenized_program in list_generator:
            for token in tokenized_program.split():
                token = token.strip()
                if token not in dict_ref:
                    dict_ref[token] = len(dict_ref)

    tldict['_pad_'] = 0
    tldict['EOF'] = 1
    tldict['-new-line-'] = 2

    for key in token_strings:
        for problem_id in token_strings[key]:
            build_dict( ( corr_prog for _, inc_prog, corr_prog in token_strings[key][problem_id]), tldict)

    # required for some programs in the test dataset.
    # 这里的33是不是有点小？
    for idx in range(33):
        if 'IDENTIFIER%d@' % idx not in tldict:
            tldict['IDENTIFIER%d@' % idx] = len(tldict)

    print 'dictionary size:', len(tldict)
    assert len(tldict) > 50
    return tldict



def generate_name_dict_store(db_path, bins):
    '''
    给定数据库的路径
    将数据库中每一个tokenzied_code所对应的name_dict和name_seq存储到一个dictionary中
    '''
    name_dict_store={}
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        for row in cursor.execute("SELECT code_id, name_dict, name_seq FROM Code;"):
            code_id = str(row[0])
            name_dict = json.loads(row[1])
            name_seq = json.loads(row[2])
            name_dict_store[code_id] = (name_dict, name_seq)
    print 'name_dict_store len:', len(name_dict_store)
    return name_dict_store


# maintain max_fix_length to keep consistentcy with deepfix.
def generate_training_data(db_path, bins, validation_users, min_program_length, max_program_length, \
                                    max_fix_length, max_mutations, max_variants, seed):
    rng = np.random.RandomState(seed)
    # 貌似这个tokenize并没有什么用
    # tokenize = C_Tokenizer().tokenize
    convert_to_new_line_format = C_Tokenizer().convert_to_new_line_format

    mutator_obj = Typo_Mutate_Java(rng)
    mutate = partial(typo_mutate, mutator_obj)

    token_strings = {'train': {}, 'validation': {}}

    exceptions_in_mutate_call = 0
    total_mutate_calls = 0
    program_lengths, fix_lengths = [], []

    # 我们是没有problem_id这一说的 但是使用code_id来代指
    code_id_list = []
    for bin_ in bins:
        for problem_id in bin_:
            code_id_list.append(problem_id)

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # query = "SELECT user_id, code_id, tokenized_code FROM Code " + "WHERE problem_id=? and codelength>? and codelength<? and errorcount=0;"
        # 原来是根据problem_id来进行选择的，但是我们并不关注problem_id，
        # 而且我们的数据库中也并没有errorcount这一说
        # 所以重新修改查询语句

        # 但总归要划分的嘛
        code_id_list = []
        query = "SELECT code_id FROM Code WHERE codelength>? and codelength<?;"
        for row in cursor.execute(query, (min_program_length, max_program_length)):
            # 得到所有的code_id
            code_id_list.append(row[0])
        
        # 我们在这里手动分割数据集，按照8:1:1的比例
        # 首先将顺序打乱
        rng.shuffle(code_id_list)

        validation_code_id_list = code_id_list[0: int(0.1 * len(code_id_list))]
        training_code_id_list = code_id_list[int(0.1 * len(code_id_list)) :]

        query = "SELECT code_id, tokenized_code FROM Code " + "WHERE codelength>? and codelength<?;"
        for row in cursor.execute(query, (min_program_length, max_program_length)):
            code_id, tokenized_program = map(str, row)
            # 确定是validation还是train
            key = 'validation' if code_id in validation_code_id_list else 'train'

            # 计算程序token的数量
            program_length = len(tokenized_program.split())
            program_lengths.append(program_length)

            if program_length >= min_program_length and program_length <= max_program_length:
                # 开始mutate

                total_mutate_calls += 1
                try:
                    iterator = mutate(tokenized_program, max_mutations, max_variants)
                except FailedToMutateException:
                    print code_id
                    exceptions_in_mutate_call += 1
                except LoopCountThresholdExceededException:
                    print code_id
                    exceptions_in_mutate_call += 1
                except ValueError:
                    print code_id
                    exceptions_in_mutate_call += 1
                    raise
                except AssertionError:
                    print code_id
                    exceptions_in_mutate_call += 1
                    raise
                except Exception:
                    print code_id
                    exceptions_in_mutate_call += 1
                    raise
                else:
                    tokenized_program = remove_empty_new_lines(convert_to_new_line_format(tokenized_program))
                    for corrupt_program, fix in iterator:
                        corrupt_program_length = len(corrupt_program.split())
                        fix_length             = len(fix.split())
                        fix_lengths.append(fix_length)
                        if corrupt_program_length >= min_program_length and \
                        corrupt_program_length <= max_program_length and fix_length <= max_fix_length:
                            corrupt_program = remove_empty_new_lines(convert_to_new_line_format(corrupt_program))
                            # 我们并没有problem_id，这里用code_id来代替
                            try:
                                token_strings[key][code_id] += [(code_id, corrupt_program, tokenized_program)]
                            except:
                                token_strings[key][code_id] = [(code_id, corrupt_program, tokenized_program)]

    program_lengths = np.sort(program_lengths)
    fix_lengths = np.sort(fix_lengths)

    print 'Statistics'
    print '----------'
    print 'Program length:  Mean =', np.mean(program_lengths), '\t95th %ile =', program_lengths[int(0.95 * len(program_lengths))]
    try:
        print 'Mean fix length: Mean =', np.mean(fix_lengths), '\t95th %ile = ', fix_lengths[int(0.95 * len(fix_lengths))]
    except Exception as e:
        print e
        print 'fix_lengths'
    print 'Total mutate calls:', total_mutate_calls
    print 'Exceptions in mutate() call:', exceptions_in_mutate_call, '\n'
    
    return token_strings, mutator_obj.get_mutation_distribution()


if __name__=='__main__':
    # maintain it to keep consistency with deepfix.
    max_program_length = 450
    min_program_length = 75
    max_fix_length = 25
    seed = 1189

    max_mutations = 5
    max_variants = 2

    db_path 		 = 'data/java_raw_data/raw_java_dataset_new.db'
    # 原来的使用validation_users来分割valiation数据集，我们不需要
    # validation_users = np.load('data/iitk-dataset/validation_users.npy', allow_pickle=True).item()
    # bin里面存储的是problem_id 我们也不需要
    # bins 			 = np.load('data/iitk-dataset/bins.npy', allow_pickle=True)
    # mock bins
    bins = get_bins(db_path, min_program_length, max_program_length)
    validation_users = {}

    # 生成的文件夹路径
    output_directory = os.path.join('data', 'network_inputs', "RLAssist-seed-%d" % (seed,))
    print 'output_directory:', output_directory
    mkdir(os.path.join(output_directory))

    # 将数据库中code的name_dict, name_seq保存在一个字典中
    name_dict_store = generate_name_dict_store(db_path, bins)

    # 生成训练集
    token_strings, mutations_distribution = generate_training_data(db_path, bins, validation_users, min_program_length,\
                            max_program_length, max_fix_length, max_mutations,\
                            max_variants, seed)



    # 将结果保存下来
    # np.save(os.path.join(output_directory, 'tokenized-examples.npy'), token_strings)
    # np.save(os.path.join(output_directory, 'error-seeding-distribution.npy'), mutations_distribution)

    tl_dict = build_dictionary(token_strings, {})
    # print tl_dict

    token_vectors = vectorize_data(token_strings, tl_dict, max_program_length)

    # print token_vectors

    save_bins(output_directory, tl_dict, token_vectors, bins, name_dict_store)
    print '\n\n---------------all outputs written to {}---------------\n\n'.format(output_directory)
