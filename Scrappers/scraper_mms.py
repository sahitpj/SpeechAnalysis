from bs4 import BeautifulSoup
from urllib.request import urlopen
import re
for count in range(1,1447): #1447 was the ID of the last speech MMS gave in the URL beneath
    speech_html =urlopen("http://archivepmo.nic.in/drmanmohansingh/speech-details.php?nodeid="+str(count)+"").read()
    speechtext = BeautifulSoup(speech_html,"html.parser")
    h2_tags=speechtext.find('h2',attrs={"class": "date"})
    if(len(re.split(r"([a-zA-z\s]+)$", h2_tags.text))==3):
        date=re.split(r"([a-zA-z\s]+)$", h2_tags.text)[0]
        location=re.split(r"([a-zA-z\s]+)$", h2_tags.text)[1]
        date2 = date.replace(' ', '_')
        date2 = date2.replace(',', '')
        location2=location.replace(' ', '_')
        filename=str(count)+"_"+date2+"_"+location2 #A filename with count, date and location
    else:
        filename=str(count)+"_"+"speech"
    p_tags=speechtext.find_all('p')
    f = open(filename+".txt",'w',encoding='utf-8')
    for body in p_tags:
        print(body.text, file=f)
        break;