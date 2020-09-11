import csv
import matplotlib.pyplot as plt
import prettytable as pt

cpct_plus = '''|        0~50        |      21767      |      19887      |      905       |       154        |     821      |
|       50~100       |      30325      |      26942      |      1554      |       561        |     1268     |
|      100~200       |      41200      |      36804      |      1682      |       1009       |     1705     |
|      200~450       |      35390      |      30925      |      1286      |       1277       |     1902     |
|      450~1000      |      16353      |      13606      |      525       |       720        |     1502     |

'''

panic_data = '''|        0~50        |      27554      |       2650      |      6009      |       2468       |    16427     |
|       50~100       |      41020      |       2858      |      5849      |       4430       |    27883     |
|      100~200       |      55272      |       4119      |      7407      |       6771       |    36975     |
|      200~450       |      47637      |       3372      |      6116      |       7044       |    31105     |
|      450~1000      |      21929      |       1405      |      2506      |       4649       |    13369     |
'''

rlassist = '''|        0~50        |      15051      |       4127      |      2641      |        0         |     8658     |
|      100~200       |      56219      |      17522      |     10951      |        0         |    29688     |
|      200~450       |      55033      |      16113      |     10472      |        0         |    30562     |
|       50~100       |      36089      |      11421      |      6745      |        0         |    19197     |
|      450~1000      |      24456      |       5776      |      3291      |        0         |    16655     |
'''

deepfix = '''|        0~50        |       8621      |       2225      |      1431      |       1533       |     3432     |
|      100~200       |      38914      |      11327      |      8776      |       5929       |    12882     |
|      200~450       |      40960      |      11986      |      9522      |       8618       |    10834     |
|       50~100       |      23212      |       6529      |      4769      |       2897       |     9017     |
|      450~1000      |      14669      |       3723      |      2872      |       4707       |     3367     |
'''

def time_data_to_csv(time_res):
    time_data = time_res.split('|\n')
    data_to_store = []
    for item in time_data:
        data = item.split('|')
        if len(data) == 3:
            data_to_store.append([data[1].strip(), data[2].strip()])


def get_single_repair_rate(tool_name, data, is_print=False):
    '''
    Only process result for one tool
    '''
    tb = pt.PrettyTable()
    tb.field_names = ['Token Range', 'Complete Repair Rate', 'Partial Repair Rate', 'Cascading Error Rate', 'Dummy Repair Rate']

    # replace ' ' with ''
    data = data.replace(' ', '')
    data = data.split('|\n')
    repair_rate_by_token = {}
    for item in data:
        data_by_token = item.split('|')[1:]
        
        try:
            complete_repair_rate = str(round(int(data_by_token[2]) * 1.00 / int(data_by_token[1]) * 100, 2)) + '%'
            partial_repair_rate  = str(round(int(data_by_token[3]) * 1.00 / int(data_by_token[1]) * 100,2)) + '%'
            cascading_error_rate = str(round(int(data_by_token[4]) * 1.00 / int(data_by_token[1]) * 100,2)) + '%'
            dummy_rapair_rate    = str(round(int(data_by_token[5]) * 1.00 / int(data_by_token[1]) * 100,2)) + '%'
            tb.add_row([data_by_token[0], complete_repair_rate, partial_repair_rate, cascading_error_rate, dummy_rapair_rate])
            repair_rate_by_token[data_by_token[0]] = [tool_name, complete_repair_rate, partial_repair_rate, cascading_error_rate, dummy_rapair_rate]

        except:
            if is_print:
                print(tb)
    
    return repair_rate_by_token

def analyze_all_tool(data_dict):
    '''
    data_dict: {'Tool Name' : tool_data }
    '''
    repair_rate_list = []
    for key in data_dict.keys():
        repair_rate = get_single_repair_rate(key, data_dict[key])
        repair_rate_list.append(repair_rate)

    analyze_complete_repair_rate(repair_rate_list, data_dict.keys())
    analyze_partial_repair_rate(repair_rate_list, data_dict.keys())
    analyze_cascading_error_rate(repair_rate_list, data_dict.keys())
    analyze_dummy_repair_rate(repair_rate_list, data_dict.keys())


    


def analyze_complete_repair_rate(repair_rate_list, name_list):
    tb = pt.PrettyTable()

    tb.field_names = ['No. Token'] + list(data_dict.keys())
    table_rows = {'0~50'  : [0]*len(name_list), \
                '50~100'  : [0]*len(name_list), \
                '100~200' : [0]*len(name_list), \
                '200~450' : [0]*len(name_list), \
                '450~1000': [0]*len(name_list)}
    for repair_rate in repair_rate_list:
        for token_range in repair_rate.keys():
            tool_name = repair_rate[token_range][0]
            index = tb.field_names.index(tool_name)
            complete_repair_rate = repair_rate[token_range][1]
            table_rows[token_range][index-1] = complete_repair_rate

    for token_range in table_rows.keys():
        tb.add_row([token_range] + table_rows[token_range])
    print('\n\n-----Complete Repair Rate-----')
    print(tb)


def analyze_partial_repair_rate(repair_rate_list, name_list):
    tb = pt.PrettyTable()

    tb.field_names = ['No. Token'] + list(data_dict.keys())
    table_rows = {'0~50'  : [0]*len(name_list), \
                '50~100'  : [0]*len(name_list), \
                '100~200' : [0]*len(name_list), \
                '200~450' : [0]*len(name_list), \
                '450~1000': [0]*len(name_list)}
    for repair_rate in repair_rate_list:
        for token_range in repair_rate.keys():
            tool_name = repair_rate[token_range][0]
            index = tb.field_names.index(tool_name)
            complete_repair_rate = repair_rate[token_range][2]
            table_rows[token_range][index-1] = complete_repair_rate

    for token_range in table_rows.keys():
        tb.add_row([token_range] + table_rows[token_range])
    print('\n\n-----Partial Repair Rate-----')
    print(tb)


def analyze_cascading_error_rate(repair_rate_list, name_list):
    tb = pt.PrettyTable()

    tb.field_names = ['No. Token'] + list(data_dict.keys())
    table_rows = {'0~50'  : [0]*len(name_list), \
                '50~100'  : [0]*len(name_list), \
                '100~200' : [0]*len(name_list), \
                '200~450' : [0]*len(name_list), \
                '450~1000': [0]*len(name_list)}
    for repair_rate in repair_rate_list:
        for token_range in repair_rate.keys():
            tool_name = repair_rate[token_range][0]
            index = tb.field_names.index(tool_name)
            complete_repair_rate = repair_rate[token_range][3]
            table_rows[token_range][index-1] = complete_repair_rate

    for token_range in table_rows.keys():
        tb.add_row([token_range] + table_rows[token_range])
    print('\n\n-----Cascading Error Rate-----')
    print(tb)

def analyze_dummy_repair_rate(repair_rate_list, name_list):
    tb = pt.PrettyTable()

    tb.field_names = ['No. Token'] + list(data_dict.keys())
    table_rows = {'0~50'  : [0]*len(name_list), \
                '50~100'  : [0]*len(name_list), \
                '100~200' : [0]*len(name_list), \
                '200~450' : [0]*len(name_list), \
                '450~1000': [0]*len(name_list)}
    for repair_rate in repair_rate_list:
        for token_range in repair_rate.keys():
            tool_name = repair_rate[token_range][0]
            index = tb.field_names.index(tool_name)
            complete_repair_rate = repair_rate[token_range][4]
            table_rows[token_range][index-1] = complete_repair_rate

    for token_range in table_rows.keys():
        tb.add_row([token_range] + table_rows[token_range])
    print('\n\n-----Dummy Repair Rate-----')
    print(tb)

def analyze_total_repair_rate(data_dict):
    for tool_name in data_dict.keys():
        print(tool_name)
        tool_data = data_dict[tool_name]
        


data_dict = {'CPCT+': cpct_plus, 'Panic': panic_data, 'RLAssist': rlassist, 'Deepfix' : deepfix}
analyze_all_tool(data_dict)

analyze_total_repair_rate(data_dict)


