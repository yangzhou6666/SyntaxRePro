import numpy as np
import os
import javac_parser
import difflib
import numpy as np
import time
import sys

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
                name_dictionary[token] = 'IDENTIFIER' + \
                    '@' + str(names.index(token) + 1)


    for token in fix.split():
        if 'IDENTIFIER' in token:
            if token not in name_dictionary:
                name_dictionary[token] = 'IDENTIFIER' + \
                    '@' + str(names.index(token) + 1)

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


    for session_id in token_strings:
        token_vectors[session_id] = []
        for prog_tokens, fix_tokens in token_strings[session_id]:
            prog_vector = vectorize(prog_tokens, tl_dict, max_program_length, drop_ids, reverse=True, vecFor='encoder')
            fix_vector = vectorize(fix_tokens, tl_dict,  max_fix_length, drop_ids, reverse=False, vecFor='decoder')

            if (prog_vector is not None) and (fix_vector is not None):
                token_vectors[session_id].append((prog_vector, fix_vector))
            else:
                skipped += 1


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
    # tl_dict['_eos_'] = 1
    tl_dict['EOF'] = 1
    tl_dict['~'] = 2

    if drop_ids:
        # tl_dict['_<id>_@'] = 3
        tl_dict['IDENTIFIER'] = 3


    for session_id in token_strings:
        build_dict((prog + ' ' + fix for prog, fix in token_strings[session_id]), tl_dict)


    assert len(tl_dict) > 4
    return tl_dict


def tokenizer(code):

    # code = "\n".join([s for s in code.splitlines() if s.strip()])

    tokens = []
    name_dict = {}
    name_seq = []
    loc = len(code.split('\n'))
    tokens_by_line = [list(str(i)) for i in range(loc)]
    for line in tokens_by_line:
        line.append('~')

    lex_res = java.lex(code)
    for lex in lex_res:
        line_num = lex[2][0]
        if lex[0] == 'IDENTIFIER':
            id = lex[1]
            name_seq.append(id)
            if id in name_dict:
                tokens_by_line[line_num - 1].append("IDENTIFIER@" + str(name_dict[id]))
            else:
                name_dict[id] = len(name_dict) + 1
                tokens_by_line[line_num - 1].append("IDENTIFIER@" + str(name_dict[id]))
        else: 
            tokens_by_line[line_num - 1].append(lex[0])
    
    token_string_by_line = []
    for line in tokens_by_line:
        tokens.append(' '.join(line))
        token_string_by_line.append(' '.join(line))

    return token_string_by_line

def tokenizer_by_token(code):

    # code = "\n".join([s for s in code.splitlines() if s.strip()])

    tokens = []
    name_dict = {}
    name_seq = []
    loc = len(code.split('\n'))
    tokens_by_line = [list(str(i)) for i in range(loc)]
    for line in tokens_by_line:
        line.append('~')

    lex_res = java.lex(code)
    for lex in lex_res:
        line_num = lex[2][0]
        if lex[0] == 'IDENTIFIER':
            id = lex[1]
            name_seq.append(id)
            if id in name_dict:
                tokens_by_line[line_num - 1].append("IDENTIFIER@" + str(name_dict[id]))
            else:
                name_dict[id] = len(name_dict) + 1
                tokens_by_line[line_num - 1].append("IDENTIFIER@" + str(name_dict[id]))
        else: 
            tokens_by_line[line_num - 1].append(lex[0])
    
    token_string_by_line = []
    for line in tokens_by_line:
        tokens.append(' '.join(line))
        token_string_by_line.append(' '.join(line))

    return ' '.join(token_string_by_line), name_dict, name_seq

def generate_fix(before, after):
    '''
    return tokenized syntactic-invalid code, and corresponding (tokenized) fixes
    tokenized syntactic-invalid code is a string
    tokenized fixes is a list of strings
    '''


    tokenized_before = tokenizer(before)
    tokenized_after = tokenizer(after)

    diff = difflib.ndiff(tokenized_before, tokenized_after)

    diffs = []
    for i in diff:
        if i.startswith('+'):
            diffs.append(i)
    return ' '.join(tokenized_before), diffs, ' '.join(tokenized_after), '-1'

def save_bins(destination, tl_dict, token_vectors):
    # to-do: real 5-folder training

    session_ids = token_vectors.keys()

    for i in range(5):
        train_list, validation_list, test_list = split_dataset(
            rng, list(session_ids), train=0.9, validation=0.05, test=0.05)
        # split train, validation and test set session_ids 
        token_vectors_this_fold = {'train': [], 'validation': [], 'test': []}

        for session_id in train_list:
            for token_pair in token_vectors[session_id]:
                token_vectors_this_fold['train'] += [token_pair]
        
        for session_id in validation_list:
            for token_pair in token_vectors[session_id]:
                token_vectors_this_fold['validation'] += [token_pair]

        for session_id in test_list:
            for token_pair in token_vectors[session_id]:
                token_vectors_this_fold['test'] += [token_pair]
            #  += token_vectors[session_id]
    
        make_dir_if_not_exists(os.path.join(destination, 'bin_%d' % i))

        save_pairs(os.path.join(destination, 'bin_%d' %
                                i), token_vectors_this_fold, tl_dict)

def get_rev_dict(dict_):
    assert len(dict_) > 0, 'passed dict has size zero'
    rev_dict_ = {}
    for key, value in dict_.items():
        rev_dict_[value] = key

    return rev_dict_

def save_dictionaries(destination, tl_dict):
    all_dicts = (tl_dict, get_rev_dict(tl_dict))
    np.save(os.path.join(destination, 'all_dicts.npy'), all_dicts)

def save_pairs(destination, token_vectors, tl_dict):
    for key in token_vectors.keys():
        np.save(os.path.join(destination, ('examples-%s.npy' % key)),
                token_vectors[key])
        save_dictionaries(destination, tl_dict)


def split_dataset(rng, session_ids, train=0.9, validation=0.05, test=0.05):
    total = len(session_ids)
    rng.shuffle(session_ids)
    test_list = session_ids[0:int(total * test)]
    validation_list = session_ids[int(total * test) + 1 : int(total * test) * 2]
    train_list = session_ids[int(total * test) * 2 + 1 : ]
    return train_list, validation_list, test_list




def generate_training_data_2(data_dir):
    token_strings = {}
    dir_list = os.listdir(data_dir)
    for senssion_dir in dir_list:
        events_dir_list = os.listdir(os.path.join(data_dir, senssion_dir))
        for events_dir in events_dir_list:
            before_path = os.path.join(data_dir, senssion_dir, events_dir, 'before.java')
            after_path = os.path.join(data_dir, senssion_dir, events_dir, 'after.java')
        
        with open(before_path) as before:
            with open(after_path) as after:
                before_code = before.read()
                after_code = after.read()
        

        corrupt_program, fixes, correct_program, dumpy_fix = generate_fix(before_code, after_code)

        if len(fixes) > 0:
            first_fix = fixes[0][2:]
        else:
            first_fix = dumpy_fix

        try:
            token_strings[senssion_dir] += [rename_ids_(rng, corrupt_program,first_fix)]
        except:
            token_strings[senssion_dir] = [rename_ids_(rng, corrupt_program,first_fix)]
    
    return token_strings

def generate_evaluation_data(rng, data_dir, data_length):
    '''
    return data format: {session_id: [(tokenzied_program, name_dict, name_sequence)]}
    '''
    token_strings = {}
    dir_list = os.listdir(data_dir)

    rng.shuffle(dir_list)
    count = 0

    for file_event in dir_list:
        if count >= data_length:
            break
        file_id = file_event.split('_')[0]

        fail_file_path = os.path.join(data_dir, file_event, '0')
        fail = open(fail_file_path)
        fail_code = fail.read()
        fail_code_without_blank_lines = "\n".join([s for s in fail_code.splitlines() if s.strip()])

        tokens, name_dict,name_seq = tokenizer_by_token(fail_code_without_blank_lines)
        dump_id = 0
        token_strings[file_id] = (tokens, name_dict,name_seq, dump_id, dump_id)
        count += 1
    
    return token_strings

def token_to_source_dict(rng, data_dir):
    token_to_source_dict = {}
    java = javac_parser.Java()
    token_strings = {}
    dir_list = os.listdir(data_dir)

    rng.shuffle(dir_list)

    count = 0
    for file_event in dir_list:
        if count >= 5000:
            break
        if count % 1000 == 0:
            print count

        fail_file_path = os.path.join(data_dir, file_event, '0')
        fail = open(fail_file_path)
        fail_code = fail.read()
        tokens = java.lex(fail_code)
        for token in tokens:
            if token[0] == 'IDENTIFIER' or token[0] == 'INTLITERAL' or token[0] == 'DOUBLELITERAL' \
                                        or token[0] == 'CHARLITERAL' or token[0] == 'FLOATLITERAL' \
                                        or token[0] == 'STRINGLITERAL' or token[0] == 'LONGLITERAL'\
                                        or token[0] == 'ERROR':
                continue
            else:
                token_to_source_dict[token[0]] = token[1]
        count += 1

    print token_to_source_dict


def generate_training_data(rng, data_dir, data_length):
    '''
    return data format: {'session_id': [(program, fix),(program, fix)]}
    '''
    token_strings = {}
    dir_list = os.listdir(data_dir)

    rng.shuffle(dir_list)

    count = 0
    for file_event in dir_list:
        if count >= data_length:
            break
        file_id = file_event.split('_')[0]

        fail_file_path = os.path.join(data_dir, file_event, '0')
        success_file_path = os.path.join(data_dir, file_event, '1')

        fail = open(fail_file_path)
        success = open(success_file_path)

        fail_code = fail.read()
        success_code = success.read()

        fail_code_without_blank_lines = "\n".join([s for s in fail_code.splitlines() if s.strip()])
        success_code_without_blank_lines = "\n".join([s for s in success_code.splitlines() if s.strip()])

        if len(fail_code_without_blank_lines.split('\n')) != len(success_code_without_blank_lines.split('\n')):
            continue

        corrupt_program, fixes, correct_program, dumpy_fix = generate_fix(fail_code_without_blank_lines, success_code_without_blank_lines)
        if len(fixes) == 0 or len(fixes) > 1:
            continue
        first_fix = fixes[0][2:]
    
        try:
            token_strings[file_id] += [rename_ids_(rng, corrupt_program,first_fix)]
        except:
            token_strings[file_id] = [rename_ids_(rng, corrupt_program,first_fix)]
        count += 1

    return token_strings



if __name__ == '__main__':
    num_train_example = int(sys.argv[1])
    print type(num_train_example)
    num_eval_example = int(sys.argv[2])

    start_time = time.time()
    java = javac_parser.Java()
    seed = 1189
    rng = np.random.RandomState(seed)

    # Here you need path to your blackbox dataset
    data_source_dir = '../../java_data'

    create_time = time.strftime('%Y-%m-%d-%H:%M:%S',time.localtime(time.time()))

    data_storage_dir = os.path.join('./java-data-storage', (create_time + '-' + str(num_train_example)))
    os.system('mkdir %s' % (data_storage_dir))
    print 'Data stored at %s' % (data_storage_dir)
    print 'Start generate %d training examples ....' % (num_train_example)

    # generate raw dataset for training, validation and testing.
    raw_training_data = generate_training_data(rng, data_source_dir, num_train_example)
    np.save(os.path.join(data_storage_dir, 'tokenized-examples.npy'), raw_training_data)

    # get tl_dict
    tl_dict = build_dictionary(raw_training_data, drop_ids = True)
    vectorized_tokens = vectorize_data(raw_training_data, tl_dict, max_program_length=450, max_fix_length=50, drop_ids=True)

    # save raw files
    save_bins(os.path.join(data_storage_dir), tl_dict, vectorized_tokens)

    # generate evaluation data
    print 'Start generate %d evaluation examples ....' % (num_eval_example)
    evaluation_data = generate_evaluation_data(rng, data_source_dir, num_eval_example)
    np.save(os.path.join(data_storage_dir, 'evaluation.npy'), evaluation_data)


    print("Time consumed: ", time.time() - start_time)
    



