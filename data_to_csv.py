'''
Author: Amol Kapoor
Description: Turns data files into csv
'''

import data_constants as dc

def get_max_len(lines):
    max_len = 0
    max_len_arr = []
    for line in lines:
        max_len = max(max_len, len(line.split(' ')))
        max_len_arr.append(len(line.split(' ')))
    max_len_arr.sort()
    return max_len

def to_csv(fdata, flabel, fname):
    fdata.seek(0)
    flabel.seek(0)

    csv = open(fname, 'w+')
    csv.write('data,labels\n')

    lines1 = fdata.readlines()
    lines2 = flabel.readlines()

    max_len_in = dc.MAX_LEN_IN
    max_len_out = dc.MAX_LEN_OUT

    for line1, line2 in zip(lines1, lines2):
        input_len = len(line1.split(' '))
        output_len = len(line2.split(' '))
        if input_len > max_len_in or output_len > max_len_out:
            continue
        spacing_in = dc.EMPT_ID * (max_len_in - input_len - 1)
        spacing_out = dc.EMPT_ID * (max_len_out - output_len - 1)
        line = '%s%s <EOS>#%s <EOS> %s\n' % (spacing_in, line1.rstrip(),
                                             line2.rstrip()[::-1], spacing_out)
        csv.write(line)

normal = open(dc.NORMAL_SENTENCE_PATH, 'r+')
simple = open(dc.SIMPLE_SENTENCE_PATH, 'r+')

to_csv(normal, simple, dc.CSV_FILE_PATH)
