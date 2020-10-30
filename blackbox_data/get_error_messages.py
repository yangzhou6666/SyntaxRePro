import javac_parser
import os
import time
import csv
import multiprocessing
import prettytable as pt

def save_to_csv(error_dict, file_path):
    with open(file_path,'w') as f:
        csv_write = csv.writer(f)
        csv_head = ['Class', 'Number']
        for x, y in error_dict.items():
            csv_write.writerow([x,y])


def classify_error(file_path_list):
    java = javac_parser.Java()
    error_dict = {}
    first_error_dict = {}
    count_dict = {}
    count = 0

    for i in file_path_list:
        count += 1
        with open(i, 'r') as f:
            source_code = f.read()
            try:
                dignostic_messages = java.check_syntax(source_code)
            except:
                continue
            if (len(dignostic_messages) == 0):
                continue
            # count the distribution of number of error messages
            message_count = 0
            for message in dignostic_messages:
                if (message[0] == 'ERROR'):
                    message_count += 1
            # count all the error messages
            if message_count in count_dict:
                count_dict[message_count] += 1
            else:
                count_dict[message_count] = 1       

            # TODO: refactoring code


            # only consider the first error message.
            if (dignostic_messages[0][0] == 'ERROR'):
                if dignostic_messages[0][1] in first_error_dict:
                    first_error_dict[dignostic_messages[0][1]] += 1
                else:
                    first_error_dict[dignostic_messages[0][1]] = 1
            

            for message in dignostic_messages:
                if (message[0] == 'ERROR'):
                    # count all the error messages
                    if message[1] in error_dict:
                        error_dict[message[1]] += 1
                    else:
                        error_dict[message[1]] = 1

    return error_dict, first_error_dict, count_dict


def find_code_by_error(error_type):
    java = javac_parser.Java()
    src_files_dir = './src_files/'
    for i in os.listdir(src_files_dir):
        with open(src_files_dir + i, 'r') as f:
            source_code = f.read()
            try:
                dignostic_messages = java.check_syntax(source_code)
            except:
                print(i)
            else:
                if (len(dignostic_messages) == 0):
                    continue
                # print(dignostic_messages)
                if dignostic_messages[0][1] == error_type:
                    # print(source_code)
                    # print('----------------------------------------------')
                    os.system('/Users/zhouyang/Downloads/error_recovery_experiment/runner/java_parser_none %s' % ())


if __name__=="__main__":
    start_time = time.time()
    java = javac_parser.Java()
    src_files_dir = './src_files/'
    java_pairs_dir = '/Volumes/ssd/SyntaxRePro_data/java_data'
    print(len(os.listdir(java_pairs_dir)))
    
    exit()

    error_dict, first_error_dict, count_dict = classify_error()
    tb = pt.PrettyTable()

    tb.field_names = ['Error Type', '# of Occurence']
    for key in error_dict.keys():
        tb.add_row([key, error_dict[key]])
    print(tb)

    file_path = './error_distribution.csv'
    save_to_csv(error_dict, file_path)


    print("Time Spent:", time.time() - start_time)

