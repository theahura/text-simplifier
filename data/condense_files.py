"""
Author: Amol Kapoor
Condenses files for wiki-rev-split database into two files.
"""

import os

def condense_files(condensed_fname, extension):
    '''
    Gets all files of an extension and condenses it to one line per file.
    '''
    file_list = []
    condensed = open(condensed_fname, 'w+')

    sorted_files = os.listdir('.')
    sorted_files.sort()

    for f_name in sorted_files:
        if f_name.endswith(extension):
            file_list.append(f_name)

    print file_list[0:10]
    print '=========================================='

    for f_name in file_list:
        f = open(f_name, 'r+')
        for line in iter(f.readline, b''):
            line = line.rstrip()
            condensed.write(line)
        condensed.write('\n')
        f.close()
    condensed.close()

condense_files('normal', '.old')
condense_files('simple', '.new')

