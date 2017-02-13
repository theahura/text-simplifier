"""
Author: Amol Kapoor
Description: Store all of the same article onto the same line
"""

import pandas

def condense_file(f, fname):
    '''
    Takes a file and moves all lines related to a single article to one line.
    '''
    aligned_doc = open(fname + "_aligned", 'w+')
    current_article = fname
    article = ""
    while True:
        # Get the line.
        line = f.readline().rstrip()
        if not line:
            break

        # Match by article and append to aligned doc.
        line = line.split('\t')
        if line[0] != current_article:
            if len(article) > 0:
                aligned_doc.write(current_article + '\t' + article + '\n')
            current_article = line[0]
            article = ""

        if len(line) < 3:
            continue
        article += line[2]
    return aligned_doc

def _get_topics(f):
    '''
    Returns a list of topics from a file.
    '''
    f_topics = {}
    while True:
        f_line = f.readline().split('\t')
        if not f_line[0]:
            break
        if len(f_line) > 1:
            f_topics[f_line[0]] = f_line[1]
    return f_topics

def remove_discrepencies(f1, f2):
    '''
    Outputs errors in the dataset alignment for user to handle.
    '''
    f1.seek(0)
    f2.seek(0)

    f1_topics = _get_topics(f1)
    f2_topics = _get_topics(f2)

    print len(f1_topics)
    print len(f2_topics)

    f1.seek(0)
    f2.seek(0)

    for topic, line in f2_topics.iteritems():
        if topic in f1_topics:
            f1.write(f1_topics[topic])
            f2.write(line)

    f1.truncate()
    f2.truncate()

    f1.seek(0)
    f2.seek(0)
    print len(f1.readlines())
    print len(f2.readlines())



normal = open('normal.txt', 'r+')
simple = open('simple.txt', 'r+')

normal = condense_file(normal, 'normal')
simple = condense_file(simple, 'simple')

remove_discrepencies(normal, simple)
