import re
import sqlite3

# LIST='test.txt'
LIST='file_list.txt'
OUTPUT='output.txt'
file_dic = {}
conn = sqlite3.connect('example.db')
o = open(OUTPUT, 'w')
with open(LIST) as f:
    for file_name  in f:
        print file_name
        file_name = file_name.strip()
        if not file_name: break
        with open(file_name) as g:
            file_dic[g.name] = {} 
            file_dic[g.name]['to_list'] = []
            for line in g:
                tokens = line.split(':')
                # tokens = tokens[:2]
                if tokens[0] != 'Date':
                    tokens = tokens[:2]
                else:
                    the_date = ':'.join(tokens[1:])
                    tokens = [tokens[0], the_date.replace(',','').strip()]
                    # print tokens

                if re.search('$^', line):
                    d = file_dic[g.name]
                    common_str = g.name + ':' + d['Date'] + ':' +  d['Message-ID'] + ':' + d['From']
                    for i,val_dic in enumerate(d['to_list']):
                        type = val_dic['type']
                        value_list = val_dic['val'].split(',')
                        for j, to_value in enumerate(value_list):
                            to_value = to_value.strip()
                            if to_value:
                                o.write(common_str + ':' + to_value + ':' + type + '\n')
                    if not len(d['to_list']):
                        o.write(common_str + '\n')
                    break
                # print 'tokens[0]: %s' %tokens[0]
                if re.search('(Date|Cc|Bcc|Message-ID|From|To)$',tokens[0]):
                    # print 'token is matching, val is %s' %tokens[0]
                    if re.search('^(To|Cc|Bcc)', tokens[0]):
                        file_dic[g.name]['to_list'].append({'type':tokens[0], 'val': tokens[1].strip()})
                    else:
                        file_dic[g.name][tokens[0]] = tokens[1].strip()
                    continue

o.close()

