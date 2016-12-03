from __future__ import division
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import RegexpTokenizer
import sys
import os
import fnmatch
import json
import xml.etree.ElementTree as etree
import math
import networkx as nx
import ast
import base64
import operator
import itertools
import nltk
import warnings
warnings.filterwarnings('ignore')
root = sys.argv[1]

def lexrank(data):
	similarity_score= defaultdict()
	Stemmer = SnowballStemmer("english",ignore_stopwords=True)
	s1=PorterStemmer()
	summary = ''

	def preprocess_words(data):
	    tokenizer = RegexpTokenizer('\w+')
	    words = [word for word in tokenizer.tokenize(data)]
	    #stop = set(stopwords.words('english'))
	    #w = [word for word in words if word not in stop]
	    #w = [Stemmer.stem(i) for i in words]
	    #return w
	    return words

	def tokenize(data):
	    s = nltk.data.load('tokenizers/punkt/english.pickle')
	    sentence = s.tokenize(data.strip())
	    s_tokens = sent_tokenize(data)
	    word_tokens = [word_tokenize(s) for s in s_tokens]
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
	        return float((numerator)/denominator)
	    except:
	        return 0.0

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
	        simval = cosine_similarity(word_tokenize(node1), word_tokenize(node2), idf)	        
	        if simval > 0.15:
	            gr.add_edge(node1, node2, weight=simval)
	    return gr

	def lexrank(graph):

	    pagerank = nx.pagerank(graph, weight='weight')
	    keys = sorted(pagerank, key=pagerank.get, reverse=False)
	    return keys

	s_tokens, word_tokens = tokenize(data)
	words = list(set(preprocess_words(data)))
	lenght=len(s_tokens)
	idf = idf(word_tokens, words, lenght)
	graph = build_graph(s_tokens, idf)
	sentences = lexrank(graph)
	write_summary = sentences[:3]
	string_summary = '\n\n'.join(str(i) for i in write_summary)
	return string_summary

def naive( content ):
	sentencesMap = dict()
	reverseSentenceMap = dict()
	processedReverseSentenceMap = dict()
	processedSentenceMap = dict()

	stemmer = PorterStemmer()
	stop = set(stopwords.words('english'))
	preSentences = content.split('.')
	paragraphs = content.split('\n\n')

	sentences = list()

	for sentence in preSentences:
	    if '\n' in sentence:
	        sentence = sentence.replace('\n',' ').strip()
	    sentences.append(sentence.strip())


	for sentence in sentences:
	    encoded = hash(sentence)
	    sentencesMap[sentence] = encoded
	    reverseSentenceMap[str(encoded)] = sentence

#	print(reverseSentenceMap)
	for key in reverseSentenceMap.keys():
	    processedReverseSentenceMap[key] = reverseSentenceMap[key]


	stemmer = PorterStemmer()
	stop = set(stopwords.words('english'))
	processedSentenceMap = sentencesMap

	# for key in processedReverseSentenceMap.keys():
	#     tokens = set(nltk.word_tokenize(processedReverseSentenceMap[key]))
	#     tokens = tokens - stop
	#     # w = [stemmer.stem(i) for i in tokens]
	#     # tokens = w
	#     processedReverseSentenceMap[key] = ' '.join(tokens)    

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
		if len(list(set(s1.split()) | set(s2.split()))) == 0:
			return 0
		return len(list(set(s1.split()) & set(s2.split())))/len(list(set(s1.split()) | set(s2.split())))

	scores = dict()
	for s1 in processedSentenceMap.values():
	    for s2 in processedSentenceMap.values():
	        if s1 != s2:
	            #scores[str(s1)] = cosineSimilarity(processedReverseSentenceMap[str(s1)], processedReverseSentenceMap[str(s2)])
	            scores[str(s1)] = sentenceSimilarity(processedReverseSentenceMap[str(s1)], processedReverseSentenceMap[str(s2)])

#	print(sentencesMap)
	summary = ''
	sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
	counter = 0
	summary = ""
	number = 7
#	print(reverseSentenceMap)
	while counter < number:
	    summary += reverseSentenceMap[sorted_scores[counter][0]]
	    summary += '. '
	    counter += 1
#	print(summary)
	return summary

def cosine_similarity(content):

	sentencesMap = dict()
	reverseSentenceMap = dict()
	processedReverseSentenceMap = dict()
	processedSentenceMap = dict()

	preSentences = content.split('.')
	paragraphs = content.split('\n\n')

	sentences = list()

	for sentence in preSentences:
	    if '\n' in sentence:
	        sentence = sentence.replace('\n',' ').strip()
	        sentences.append(sentence.strip())

	for sentence in sentences:
	    encoded = hash(sentence)
	    sentencesMap[sentence] = encoded
	    reverseSentenceMap[str(encoded)] = sentence

	for key in reverseSentenceMap.keys():
	    processedReverseSentenceMap[key] = reverseSentenceMap[key]


	stemmer = PorterStemmer()
	stop = set(stopwords.words('english'))
	processedSentenceMap = sentencesMap

	for key in processedReverseSentenceMap.keys():
	    tokens = set(nltk.word_tokenize(processedReverseSentenceMap[key]))
	    # tokens = tokens - stop
	    # processedReverseSentenceMap[key] = ' '.join(tokens)
	    w = [stemmer.stem(i) for i in tokens]	    	    
	    processedReverseSentenceMap[key] = ' '.join(w)	

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

	edgeList = open("edgeList.txt", "w")
	for s1 in processedSentenceMap.values():
	    for s2 in processedSentenceMap.values():
	        if s1 != s2:
	            edgeList.write(str(s1))
	            edgeList.write(' ')
	            edgeList.write(str(s2))
	            edgeList.write(' ')
	            #edgeList.write(str(sentenceSimilarity(processedReverseSentenceMap[str(s1)], processedReverseSentenceMap[str(s2)])))
	            edgeList.write(str(cosineSimilarity(processedReverseSentenceMap[str(s1)], processedReverseSentenceMap[str(s2)])))
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
	counter = 0
	summary = ""
	while counter < number:
	    summary += reverseSentenceMap[sorted_scores[counter][0]]
	    summary += '. '
	    counter += 1

	return summary

count = 0
skipped = 0

for root, directory , files in os.walk(root):               
    for items in files:
        fullPath = os.path.join(root,items)           
        fullPath.lower()
        if "\docs\\" in fullPath:
        	print(items)  
        	# if "LA" in fullPath :
        	# 	print("skipping")
        	# 	skipped += 1
        	# 	continue    
        	if "FBI" in fullPath :
        		print("skipping")
        		skipped += 1
        		continue   
        	# naive
        	#2001 training
        	# if items == "WSJ911011-0071.xml" or items== "WSJ890817-0069.xml" or items=="WSJ911224-0095.xml" or items=="WSJ911004-0115.xml" or items== "WSJ911016-0124.xml" or items == "WSJ900426-0110.xml" or items == "WSJ871019-0079.xml" or items == "WSJ910822-0021.xml" or items == "AP900323-0036.xml" or items == "WSJ920214-0040.xml":
        	# 	print("skipping")
        	# 	skipped += 1
        	# 	continue
        	# #2001 test
        	# if items == "AP891116-0191.xml" or items == "WSJ870123-0101.xml" or items=="AP890719-0225.xml" or items=="WSJ890828-0011.xml" or items == "AP880901-0052.xml" or items=="AP890801-0025.xml" or items=="WSJ900418-0193.xml" or items=="AP881222-0089.xml" or items== "AP900323-0036.xml":
        	# 	print("skipping "+fullPath)
        	# 	skipped += 1
        	# 	continue   


        	# cosine
			# training 2001	
        	# if items == "SJMN91-06338157.xml" or items == "SJMN91-06130055.xml" or items == "AP890917-0009.xml" or items == "WSJ880520-0126.xml" or items == "AP880316-0061.xml" or items=="AP881017-0098.xml" or items=="AP881109-0149.xml" or items=="SJMN91-06052157.xml" or items=="AP891005-0230.xml" or items == "WSJ911213-0029.xml" or items=="FT921-3939.xml" or items=="FT921-6336.xml" or items =="SJMN91-06271061.xml" or items=="WSJ870804-0097.xml" or items=="AP880911-0024.xml" or items == "SJMN91-06317223.xml" :
        	# 	print("skipping")
        	# 	skipped += 1
        	# 	continue   

        	# if "LA" in fullPath:
        	# 	print("skipping")
        	# 	skipped += 1
        	# 	continue           

        	#test 2001
        	# if items=="FT923-5797.xml" or items=="FT923-5835.xml" or items=="FT921-9310.xml" or items=="FT931-3883.xml" or items =="FT941-1547.xml" or items=="FT941-575.xml" or items=="AP890403-0123.xml" or items=="FT931-11394.xml" or items=="FT934-11014.xml" or items=="SJMN91-06290146.xml" or items=="WSJ910304-0002.xml" or items=="AP880217-0175.xml" or items == "AP900512-0038.xml" or items=="WSJ910529-0003.xml" or items=="AP890722-0081.xml" or items=="FT922-8860.xml" or items=="AP900625-0160.xml" or items=="WSJ910628-0109.xml" or items =="AP880816-0234.xml" or items=="WSJ910718-0143.xml" or items=="WSJ920103-0037.xml" or items=="SJMN91-06136305.xml" or items=="SJMN91-06212161.xml":
        	# 	print("skipping "+fullPath)
        	# 	skipped += 1
        	# 	continue
        	# if "LA" in fullPath:
        	# 	print("skipping")
        	# 	skipped += 1
        	# 	continue 

        	count += 1
        	tree = etree.parse(fullPath)        	
        	docno = tree.find( "DOCNO" ).text.strip()
        	#print(docno)        	
        	# print("\nContent\n")
        	content = ""
        	contentList = tree.findall( "TEXT" )
        	for item in contentList:
        		content += item.text  
        	
        	if not content :
        		skipped += 1
        		continue

        	out_naive = "F:/system_naive/2001/training/cosine/stem/"+docno+"_naive.txt"	
        	summary = ''
        	summary = naive(content)        	
        	f = open( out_naive, 'w')
        	print(summary)
        	f.write(summary)
        	f.close()

        	# out_cosine = "F:/system_cosine/2001/cosine/test/stopwords/"+docno+"_cosine.txt"
        	# summary = cosine_similarity(content)
        	# f = open( out_cosine, 'w')
        	# f.write(summary)
        	# f.close()  

        	# out_lex = "F:/system_lex/2001/test/unprocessed/"+docno+"_lex.txt"
        	# summary = lexrank(content)
        	# f = open( out_lex, 'w')
        	# f.write(summary)
        	# f.close()  

print(count - skipped)
print(skipped)        	

print("\n\n\n\ndone!!")        	
