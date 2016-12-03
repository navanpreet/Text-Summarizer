import json
from bs4 import BeautifulSoup



soup = BeautifulSoup(open("WSJ910412-0119.xml"), "xml")
#
# print(soup.prettify())

doc_tag = soup.DOC

#print doc_tag
docno=[]
for child in doc_tag.find_all('DOCNO'):
    docno.append(child.string)
print docno



lpara=[]
for child in doc_tag.find_all('LP'):
    lpara.append(child.string)
print lpara


head=[]
for child in doc_tag.find_all('HL'):
    head.append(child.string)
print head

# memo=[]
# for child in doc_tag.find_all('MEMO'):
#     memo.append(child.string)
# print memo

text=[]
for child in doc_tag.find_all('TEXT'):
    text.append(child.string)
print text



xml_file = {"DOCNO":docno,

        "LEADPARA": lpara,

        "HEADLINE": head,

        "TEXT":text}

with open("WSJ910412-0119.json", "w") as json_file:
    json.dump(xml_file, json_file)







