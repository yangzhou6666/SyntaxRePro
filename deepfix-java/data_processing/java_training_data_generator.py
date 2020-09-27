#encoding:  utf-8
import os
from util.helpers import get_rev_dict, make_dir_if_not_exists as mkdir
import os, time, argparse, sqlite3, json, numpy as np
from util.c_tokenizer import C_Tokenizer
from functools import partial
from data_processing.typo_mutator import LoopCountThresholdExceededException, FailedToMutateException, Typo_Mutate, typo_mutate, Typo_Mutate_Java

class FixIDNotFoundInSource(Exception):
    pass

def make_dir_if_not_exists(path):
    try:
        os.makedirs(path)
    except:
        pass

def rename_ids_(rng, corrupt_program, fix):
    corrupt_program_new = ''
    fix_new = ''

    names = []
    for token in corrupt_program.split():
        if 'IDENTIFIER' in token:
            if token not in names:
                names.append(token)
    

    for token in fix.split():
        if 'IDENTIFIER' in token:
            if token not in names:
                names.append(token)
    rng.shuffle(names)
    name_dictionary = {}

    for token in corrupt_program.split():
        if 'IDENTIFIER' in token:
            if token not in name_dictionary:
                name_dictionary[token] = 'IDENTIFIER' + str(names.index(token) + 1) + '@'


    for token in fix.split():
        if 'IDENTIFIER' in token:
            if token not in name_dictionary:
                name_dictionary[token] = 'IDENTIFIER' + str(names.index(token) + 1) + '@'

    # Rename
    for token in corrupt_program.split():
        if 'IDENTIFIER' in token:
            corrupt_program_new += name_dictionary[token] + " "
        else:
            corrupt_program_new += token + " "

    for token in fix.split():
        if 'IDENTIFIER' in token:
            fix_new += name_dictionary[token] + " "
        else:
            fix_new += token + " "

    return (corrupt_program_new, fix_new)

def vectorize(tokens, tl_dict, max_vector_length, drop_ids, reverse, vecFor='encoder'):
    assert vecFor == 'encoder' or not reverse, 'reverse passed as True for decoder sequence'

    vec_tokens = []
    for token in tokens.split():
        if drop_ids and 'IDENTIFIER' in token:
            token = 'IDENTIFIER'

        try:
            vec_tokens.append(tl_dict[token])
        except Exception:
            raise

    if len(vec_tokens) > max_vector_length:
        return None

    if reverse:
        vec_tokens = vec_tokens[::-1]

    return vec_tokens


def vectorize_data(token_strings, tl_dict, max_program_length, max_fix_length, drop_ids):
    token_vectors = {}
    skipped = 0

    for key in token_strings:
        token_vectors[key] = {}
        for problem_id in token_strings[key]:
            token_vectors[key][problem_id] = []

    for key in token_strings:
        for problem_id in token_strings[key]:
            for prog_tokens, fix_tokens in token_strings[key][problem_id]:
                prog_vector = vectorize(
                    prog_tokens, tl_dict, max_program_length, drop_ids, reverse=True, vecFor='encoder')
                fix_vector = vectorize(
                    fix_tokens, tl_dict,  max_fix_length, drop_ids, reverse=False, vecFor='decoder')

                if (prog_vector is not None) and (fix_vector is not None):
                    token_vectors[key][problem_id].append(
                        (prog_vector, fix_vector))
                else:
                    print 'something not right'
                    skipped += 1

    print 'skipped during vectorization:', skipped
    return token_vectors

def build_dictionary(token_strings, drop_ids, tl_dict={}):

    def build_dict(list_generator, dict_ref):
        for tokenized_code in list_generator:
            for token in tokenized_code.split():
                if drop_ids and 'IDENTIFIER' in token:
                    continue
                token = token.strip()
                if token not in dict_ref:
                    dict_ref[token] = len(dict_ref)

    tl_dict['_pad_'] = 0
    tl_dict['EOF'] = 1
    tl_dict['~'] = 2

    if drop_ids:
        tl_dict['IDENTIFIER'] = 3


    if type(token_strings) == list:
        token_strings_list = token_strings

        for token_strings in token_strings_list:
            for key in token_strings:
                for problem_id in token_strings[key]:
                    build_dict((prog + ' ' + fix for prog,
                                fix in token_strings[key][problem_id]), tl_dict)
    else:
        for key in token_strings:
            for problem_id in token_strings[key]:
                build_dict((prog + ' ' + fix for prog,
                            fix in token_strings[key][problem_id]), tl_dict)

    print 'dictionary size:', len(tl_dict)

    assert len(tl_dict) > 4
    return tl_dict

def save_dictionaries(destination, tldict):
    all_dicts = (tldict, get_rev_dict(tldict))
    np.save(os.path.join(destination, 'all_dicts.npy'), all_dicts)

def save_pairs(destination, token_vectors, tl_dict):
    for key in token_vectors.keys():
        np.save(os.path.join(destination, ('examples-%s.npy' % key)),
                token_vectors[key])
        save_dictionaries(destination, tl_dict)


def get_bins(db_path, min_program_length, max_program_length):
    bins = []
    with sqlite3.connect(db_path) as conn:
        code_id_list = []

        cursor = conn.cursor()
        query = "SELECT code_id FROM Code WHERE codelength>? and codelength<?;"
        for row in cursor.execute(query, (min_program_length, max_program_length)):
            # get all the code_id
            code_id_list.append(row[0])
        
        length = len(code_id_list)  # total length
        n = 1 # seperate them into n directory， set to 1 not not sperate
        step = int(length / n) + 1 
        for i in range(0, length, step):
            bins.append(code_id_list[i: i + step])

    return bins

def save_bins(destination, tl_dict, token_vectors, bins):
    full_list = []

    for bin_ in bins:
        for problem_id in bin_:
            full_list.append(problem_id)

    for i, bin_ in enumerate(bins):

        token_vectors_this_fold = {'train': [], 'validation': [], 'test': []}
        # bin_ contains all the code id in this folder
        for problem_id in bin_:
            if problem_id in token_vectors['train']:
                token_vectors_this_fold['train'] += token_vectors['train'][problem_id]
            if problem_id in token_vectors['validation']:
                token_vectors_this_fold['validation'] += token_vectors['validation'][problem_id]
            if problem_id in token_vectors['validation']:
                token_vectors_this_fold['test'] += token_vectors['validation'][problem_id]

        make_dir_if_not_exists(os.path.join(destination, 'bin_%d' % i))

        print "Fold %d: Train:%d Validation:%d Test:%d" % (i, len(token_vectors_this_fold['train']),
                                                           len(token_vectors_this_fold['validation']), 
                                                           len(token_vectors_this_fold['test']))

        save_pairs(os.path.join(destination, 'bin_%d' % i), 
                token_vectors_this_fold, tl_dict)


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

    mutator_obj = Typo_Mutate_Java(rng)
    mutate = partial(typo_mutate, mutator_obj)

    rename_ids = partial(rename_ids_, rng)

    token_strings = {'train': {}, 'validation': {}, 'test': {}}

    exceptions_in_mutate_call = 0
    total_mutate_calls = 0
    program_lengths, fix_lengths = [], []

    code_id_list = []
    for bin_ in bins:
        for problem_id in bin_:
            code_id_list.append(problem_id)

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        code_id_list = []
        query = "SELECT code_id FROM Code WHERE codelength>? and codelength<?;"
        for row in cursor.execute(query, (min_program_length, max_program_length)):
            # get all the code_id
            code_id_list.append(row[0])
        

        rng.shuffle(code_id_list)

        # split into train, valiation and test test: 80%, 10%, 10%
        validation_code_id_list = code_id_list[0: int(0.1 * len(code_id_list))]
        test_code_id_list =  code_id_list[int(0.1 * len(code_id_list)): int(0.1 * len(code_id_list)) * 2]
        training_code_id_list = code_id_list[int(0.1 * len(code_id_list)) * 2 :]

        # make sure they do not intersect
        assert list(set(training_code_id_list) & set(validation_code_id_list)) == []
        assert list(set(training_code_id_list) & set(test_code_id_list)) == []
        assert list(set(validation_code_id_list) & set(test_code_id_list)) == []

        query = "SELECT code_id, tokenized_code, codelength FROM Code " + "WHERE codelength>? and codelength<?;"
        total_variant_cnt = 0
        for row in cursor.execute(query, (min_program_length, max_program_length)):
            code_id = row[0]
            tokenized_program = row[1]

            if code_id in validation_code_id_list:
                key = 'validation'
            if code_id in test_code_id_list:
                key = 'test'
            if code_id in training_code_id_list:
                key = 'train'

            # number of tokens
            program_length = row[2] # row[2] is codelength
            program_lengths.append(program_length)

            if program_length > min_program_length and program_length < max_program_length:
                # start to mutate

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
                    for corrupt_program, fix in iterator:
                        corrupt_program_length = len(corrupt_program.split())
                        fix_length             = len(fix.split())
                        fix_lengths.append(fix_length)
                        
                        if corrupt_program_length >= min_program_length and \
                        corrupt_program_length <= max_program_length and fix_length <= max_fix_length:
                            total_variant_cnt += 1
                            try:
                                corrupt_program, fix = rename_ids(corrupt_program, fix)
                            except FixIDNotFoundInSource:
                                exceptions_in_mutate_call += 1

                            try:
                                token_strings[key][code_id] += [
                                    (corrupt_program, fix)]
                            except:
                                token_strings[key][code_id] = [
                                    (corrupt_program, fix)]
    
    for key in token_strings.keys():
        print len(token_strings[key])
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
    print 'Total vairants generated: ', total_variant_cnt
    print 'Exceptions in mutate() call:', exceptions_in_mutate_call, '\n'
    
    return token_strings, mutator_obj.get_mutation_distribution()


if __name__=='__main__':
    # maintain it to keep consistency with deepfix.

    drop_ids = True
    # set drop_ids to True, since we only focus on sytactic structure
    max_program_length = 450
    min_program_length = 100
    max_fix_length = 25
    seed = 1189

    max_mutations = 5
    max_variants = 5

    db_path 		 = 'data/java_data/java_data.db'
    bins = get_bins(db_path, min_program_length, max_program_length)
    validation_users = {}

    # path to store data
    output_directory = os.path.join('data', 'network_inputs', "Deepfix-Java-seed-%d" % (seed,))
    print 'output_directory:', output_directory
    mkdir(os.path.join(output_directory))

    # store name_dict, name_seq data
    name_dict_store = generate_name_dict_store(db_path, bins)

    # generate dataset
    token_strings, mutations_distribution = generate_training_data(db_path, bins, validation_users, min_program_length,\
                            max_program_length, max_fix_length, max_mutations,\
                            max_variants, seed)

    # store
    np.save(os.path.join(output_directory, 'tokenized-examples.npy'), token_strings)
    np.save(os.path.join(output_directory, 'error-seeding-distribution.npy'), mutations_distribution)

    tl_dict = build_dictionary(token_strings, drop_ids, {})

    token_vectors = vectorize_data(token_strings, tl_dict, max_program_length, max_fix_length, drop_ids=True)


    save_bins(output_directory, tl_dict, token_vectors, bins)

    print '\n\n---------------all outputs written to {}---------------\n\n'.format(output_directory)
