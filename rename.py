import sys
import os
import fnmatch
import json

root = sys.argv[1]
count = ''

for root, directory , files in os.walk(root):        
    for items in files:
        fullPath = os.path.join(root,items)           
        fullPath.lower()       
        print(items)
        fp = open(fullPath,'r')
        content = fp.read()
        c = content.split('.')
        count += items + " " + str(len(c)) + '\n'

out_naive = "TrainRefCount.txt"	        	        	        
f = open( out_naive, 'w')        	
f.write(count)