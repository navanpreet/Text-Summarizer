file = open("C:/Users/navan/Google Drive/DUC dataset/DUC2001_Summarization_Documents/DUC2001_Summarization_Documents/data/training/d01a/d01aa/perdocs")
bigList = file.read().split('</SUM>')
smallLists = list()
names = list()
for list in bigList:
    smallList = list.split('>')
    smallLists.append(smallList)
    for l in smallList:
        if "DOCREF" in l:
            el = l.split('DOCREF="')
            for e in el:
                if "SELECTOR" in e:
                    n = e.split('"')
                    names.append(n[0]+"_ref.txt")

for i in range(len(smallLists)-1):
    file = open(names[i],'w')
    file.write(smallLists[i][1])