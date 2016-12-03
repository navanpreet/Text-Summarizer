import json
from bs4 import BeautifulSoup
import re, cgi


soup = BeautifulSoup(open("LA122989-0046.xml"), "xml")
#
# print(soup.prettify())

doc_tag = soup.DOC

#print doc_tag
docno=[]
for child in doc_tag.find_all('DOCNO'):
    docno.append(child.string)
print docno



head=[]
head_tag=doc_tag.HEADLINE.find_all(string=True)
print head_tag
head.append(head_tag)



desc=[]
sub_tag=doc_tag.SUBJECT.find_all(string=True)
# for child in doc_tag.find_all('SUBJECT'):
desc.append(sub_tag)
#print desc


text=[]
text_tag= doc_tag.TEXT.find_all(string=True)
# for child in doc_tag.find_all('TEXT'):
text.append(text_tag)
#print text




xml_file = {"DOCNO":docno,
        "DESCRIPT": sub_tag,


        "HEADLINE": head_tag,

        "TEXT":text_tag}

with open("LA122989-0046.json", "w") as json_file:
    json.dump(xml_file, json_file)







