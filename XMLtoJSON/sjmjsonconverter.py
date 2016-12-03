import json
from bs4 import BeautifulSoup



soup = BeautifulSoup(open("SJMN91-06154062.xml"), "xml")
#
# print(soup.prettify())

doc_tag = soup.DOC

#print doc_tag
docno=[]
for child in doc_tag.find_all('DOCNO'):
    docno.append(child.string)
print docno

desc=[]
for child in doc_tag.find_all('DESCRIPT'):
    desc.append(child.string)
print desc

lpara=[]
for child in doc_tag.find_all('LEADPARA'):
    lpara.append(child.string)
print lpara

section=[]
for child in doc_tag.find_all('SECTION'):
    section.append(child.string)
print section

head=[]
for child in doc_tag.find_all('HEADLINE'):
    head.append(child.string)
print head

memo=[]
for child in doc_tag.find_all('MEMO'):
    memo.append(child.string)
print memo

text=[]
for child in doc_tag.find_all('TEXT'):
    text.append(child.string)
print text


xml_file = {"DOCNO":docno,
        "DESCRIPT": desc,
        "LEADPARA": lpara,
        "SECTION": section,
        "HEADLINE": head,
        "MEMO": memo,
        "TEXT":text}

with open("SJMN91-06154062.json", "w") as json_file:
    json.dump(xml_file, json_file)







