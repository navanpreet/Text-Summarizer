import sys
import os
import fnmatch
import json

root = sys.argv[1]

for root, directory , files in os.walk(root):    
# for name in directory:
#     fullPath = os.path.join(root, name)        
#     if "docs" in fullPath:
#         print(fullPath)            
    for items in files:
        fullPath = os.path.join(root,items)           
        fullPath.lower()        
        if "perdocs" in fullPath:
        	print(fullPath)
	        file = open(fullPath,"r")
	        bigList = file.read().split('</SUM>')
	        smallLists = list()
	        names = list()
	        for item in bigList:
	        	smallList = item.split('>')
	        	smallLists.append(smallList)
	        	for l in smallList:
	        		if "DOCREF" in l:
	        			el = l.split('DOCREF="')
	        			for e in el:
	        				if "SELECTOR" in e:
	        					n = e.split('"')
	        					names.append(n[0]+"_ref.txt")
	        for i in range(len(smallLists)-1):
	        	filename = "F:/refernce/2001/test/"+names[i]
	        	f = open( filename , 'w' )
	        	f.write(smallLists[i][1])			    