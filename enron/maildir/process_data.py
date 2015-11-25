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
                tokens = tokens[:2]
                if re.search('$^', line):
                    d = file_dic[g.name]
                    common_str = g.name + ':' + d['Message-ID'] + ':' + d['From']
                    for i,val_dic in enumerate(d['to_list']):
                        type = val_dic['type']
                        value = val_dic['val']
                        if value.strip():
                            o.write(common_str + ':' + type + '=' + value + '\n')
                    if not len(d['to_list']):
                        o.write(common_str)
                    break
                # print 'tokens[0]: %s' %tokens[0]
                if re.search('(Cc|Bcc|Message-ID|From|To|X-To|X-cc|X-bcc)$',tokens[0]):
                    # print 'token is matching, val is %s' %tokens[0]
                    if re.search('(.*To|X.*|Cc|Bcc)', tokens[0]):
                        file_dic[g.name]['to_list'].append({'type':tokens[0], 'val': tokens[1].strip()})
                    else:
                        file_dic[g.name][tokens[0]] = tokens[1].strip()
                    continue

o.close()

