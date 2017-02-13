"""
Author: Amol Kapoor
Description: cleans sentence aligned data
"""

def clean_file(f, fname):
    f.seek(0)
    clean = open(fname + "_aligned", 'w+')
    lines = f.readlines()
    for line in lines:
        line = line.split('\t')[2]
        clean.write(line)

normal = open('normal.aligned', 'r+')
simple = open('simple.aligned', 'r+')

clean_file(normal, 'normal')
clean_file(simple, 'simple')

