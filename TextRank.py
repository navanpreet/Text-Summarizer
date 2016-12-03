from collections import defaultdict
import math
import networkx as nx
import operator
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
# nltk.download()
sentencesMap = dict()
reverseSentenceMap = dict()
processedReverseSentenceMap = dict()
processedSentenceMap = dict()


file = open("C:/Users/navan/PycharmProjects/NLP_Assignment_2/text 1.txt", encoding="UTF-8")
f = file.read()
preSentences = f.split('.')
paragraphs = f.split('\n\n')

sentences = list()

for sentence in preSentences:
    if '\n' in sentence:
        sentence = sentence.replace('\n',' ').split()
        sentences.append(sentence).split()

for sentence in sentences:
    encoded = hash(sentence)
    sentencesMap[sentence] = encoded
    reverseSentenceMap[str(encoded)] = sentence

print(reverseSentenceMap)
for key in reverseSentenceMap.keys():
    processedReverseSentenceMap[key] = reverseSentenceMap[key]


stemmer = PorterStemmer()
stop = set(stopwords.words('english'))
processedSentenceMap = sentencesMap

for key in processedReverseSentenceMap.keys():
    tokens = set(nltk.word_tokenize(processedReverseSentenceMap[key]))
    tokens = tokens - stop
    processedReverseSentenceMap[key] = ' '.join(tokens)

# print(processedReverseSentenceMap)
# print(processedSentenceMap)
# print(reverseSentenceMap)
# print(sentencesMap)

def cosineSimilarity(s1, s2):
    map1 = defaultdict(int)
    map2 = defaultdict(int)
    for word in s1.split( ):
        map1[word.lower()] += 1
    for word in s2.split( ):
        map2[word.lower()] += 1
    commonWords = map1.keys() & map2.keys()
    dotProduct = 0
    for word in commonWords:
        dotProduct += (map1[word]*map2[word])
    term1 = 0
    for word in map1.keys():
        term1 += (map1[word]**2)
    term2 = 0
    for word in map2.keys():
        term2 += (map2[word]**2)
    denominator = math.sqrt(term1)*math.sqrt(term2)
    if denominator != 0:
        return dotProduct/denominator
    else:
        return 0


def sentenceSimilarity(s1, s2):
    return len(list(set(s1.split()) & set(s2.split())))/len(list(set(s1.split()) | set(s2.split())))

print(processedReverseSentenceMap)
print(processedSentenceMap)
edgeList = open("edgeList.txt", "w")
for s1 in processedSentenceMap.values():
    for s2 in processedSentenceMap.values():
        if s1 != s2:
            edgeList.write(str(s1))
            edgeList.write(' ')
            edgeList.write(str(s2))
            edgeList.write(' ')
            edgeList.write(str(sentenceSimilarity(processedReverseSentenceMap[str(s1)], processedReverseSentenceMap[str(s2)])))
            edgeList.write('\n')

fh = open('edgeList.txt', 'rb')
output = open('el.txt', 'w')
G = nx.read_weighted_edgelist(fh)
pr = nx.pagerank(G, alpha=0.85)
for key in pr.keys():
	output.write(str(key))
	output.write("=")
	output.write(str(pr[key]))
	output.write("\n")
output.close()
f = open('el.txt')
number = 5
topSentences = list()
scores = dict()
allSentences = f.read().split('\n')
for sent in allSentences:
    s = sent.split("=")
    if len(s) == 1:
        continue
    else:
        scores[s[0]] = s[1]

sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
print(sorted_scores)
counter = 0
summary = ""
print(reverseSentenceMap)
while counter < number:
    summary += reverseSentenceMap[sorted_scores[counter][0]]
    summary += '. '
    counter += 1

print(summary)
