from __future__ import division
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from _collections import defaultdict
from nltk.stem.snowball import SnowballStemmer
import nltk
import networkx as nx
import math
import json
import itertools
import sys
import io

reload(sys)
sys.setdefaultencoding('utf8')


similarity_score= defaultdict()

Stemmer = SnowballStemmer("english",ignore_stopwords=True)
# s1=PorterStemmer()



def preprocess_words(data):
    tokenizer = RegexpTokenizer('\w+')
    words = [word for word in tokenizer.tokenize(data)]
    stop = set(stopwords.words('english'))
    w = [word for word in words if word not in stop]
    words= [Stemmer.stem(i) for i in w]

    return words

def tokenize(data):
    s = nltk.data.load('tokenizers/punkt/english.pickle')
    sentence = s.tokenize(data.strip())
    s_tokens = sent_tokenize(data)
    word_tokens = [word_tokenize(s) for s in s_tokens]

    # print sent_tokens
    # print word_tokens
    return s_tokens, word_tokens


def idf(tokens, words,length):
    d = {}
    for word in words:
        for sent in tokens:
            if word in sent:
                if word in d.keys():
                    d[word] += 1

                else:
                    d[word] = 1

    for word, count in d.items():
        d[word] = math.log((length/float(count)))
    with open("tf_idf_scores.txt","w") as f:
        json.dump(d, f)
    return d


def cosine_similarity(s1, s2, idf):
    try:
        numerator = 0
        total_word = s1 + s2
        for word in total_word:
            s1count= s1.count(word)
            # print s1count
            s2count= s2.count(word)
            numerator += int(s1count) * int(s2count) * float((idf[word] ** 2))
            a = 0
            b = 0

        for word in s1:
            freq = s1.count(word)
            a += int(freq) * float(idf[word])
        for word in s2:
            freq = s2.count(word)
            b += int(freq) * float(idf[word])
        denominator = (math.sqrt((a**2))) * (math.sqrt((b**2)))
        return float(numerator)/denominator
    except:
        return


def adjacency_matrix(tokens, idf):
    matrix = []
    for sent1 in tokens:
        row = []
        for sent2 in tokens:
            score = cosine_similarity(sent1, sent2, idf)
            row.append(score)
        matrix.append(row)
    return matrix


def build_graph(nodes, idf):
    gr = nx.Graph()
    gr.add_nodes_from(nodes)
    nodelist = list(itertools.combinations(nodes, 2))
    for x in nodelist:
        node1 = x[0]
        node2 = x[1]
        measure = cosine_similarity(word_tokenize(node1), word_tokenize(node2), idf)
        if measure > 0.15:
            gr.add_edge(node1, node2, weight=measure)

    return gr



def lexrank(graph):

    pagerank = nx.pagerank(graph, weight='weight')
    keys = sorted(pagerank, key=pagerank.get, reverse=False)
    return keys



if __name__ == "__main__":
    summary = ''
    with io.open(sys.argv[1], "r",encoding='UTF8',errors='ignore') as fp:
        data = fp.read()
    s_tokens, word_tokens = tokenize(data)
    words = list(set(preprocess_words(data)))
    lenght=len(s_tokens)
    idf = idf(word_tokens, words, lenght)
    graph = build_graph(s_tokens, idf)
    sentences = lexrank(graph)
    write_summary = sentences[:3]

    string_summary = '\n\n'.join(str(i) for i in write_summary)
    print string_summary
    # for i in write_summary:
    #     summary += i + "\n"
    # print summary

    with open(sys.argv[2], 'w+') as f:
        f.write(string_summary)



