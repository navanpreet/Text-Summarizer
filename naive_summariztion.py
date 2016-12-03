from collections import defaultdict
import math
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
import operator

sentencesMap = dict()
reverseSentenceMap = dict()
processedReverseSentenceMap = dict()
processedSentenceMap = dict()

stemmer = PorterStemmer()
stop = set(stopwords.words('english'))

file = open("C:/Users/navan/PycharmProjects/NLP_Assignment_2/text 1.txt", encoding="UTF-8")
f = file.read()
preSentences = f.split('.')
paragraphs = f.split('\n\n')

sentences = list()

for sentence in preSentences:
    if '\n' in sentence:
        sentence = sentence.replace('\n',' ').strip()
    sentences.append(sentence.strip())


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

scores = dict()
for s1 in processedSentenceMap.values():
    for s2 in processedSentenceMap.values():
        if s1 != s2:
            scores[str(s1)] = sentenceSimilarity(processedReverseSentenceMap[str(s1)], processedReverseSentenceMap[str(s2)])

print(sentencesMap)
summary = ''
sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
counter = 0
summary = ""
number = 7
print(reverseSentenceMap)
while counter < number:
    summary += reverseSentenceMap[sorted_scores[counter][0]]
    summary += '. '
    counter += 1
print(summary)