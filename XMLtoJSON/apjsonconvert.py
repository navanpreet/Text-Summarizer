import json
from bs4 import BeautifulSoup



soup = BeautifulSoup(open("AP900622-0025.xml"), "xml")
#
# print(soup.prettify())

doc_tag = soup.DOC

#print doc_tag
docno=[]
for child in doc_tag.find_all('DOCNO'):
    docno.append(child.string)
print docno


head=[]
for child in doc_tag.find_all('HEAD'):
    head.append(child.string)
print head


text=[]
for child in doc_tag.find_all('TEXT'):
    text.append(child.string)
print text



xml_file = {"DOCNO":docno,

        "HEADLINE": head,

        "TEXT":text}

with open("AP900622-0025.json", "w") as json_file:
    json.dump(xml_file, json_file)







