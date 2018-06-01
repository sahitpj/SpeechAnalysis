from lxml import html
import requests
from urllib.request import urlopen
import re
import time
from selenium import webdriver
chrome_path=r"/usr/local/share/chromedriver"
driver=webdriver.Chrome(chrome_path)
listofweblinks=[]
driver.get("http://www.pmindia.gov.in/en/tag/pmspeech/")
for i in range(0,90):
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(3)
post_elems = driver.find_element_by_class_name("news-holder")
li_elems= post_elems.find_elements_by_tag_name("li")
for item in li_elems:
    link=item.find_element_by_css_selector('a').get_attribute('href')
    listofweblinks.append(link)

newlistofweblinks=listofweblinks[::-1] #Reverse the order of the links so they can be numbered from the past
index=0
for count in range(index,len(newlistofweblinks)):
    print(count)
    page=requests.get(newlistofweblinks[count])
    speechtext = html.fromstring(page.content)
    h2_tag = speechtext.find('.//h2')
    if h2_tag or h2_tag.text is None:
        headline="None"
    else: 
        headline=h2_tag.text
        headline = headline.replace('“','')
        headline = headline.replace('”','')
        headline = headline.replace('’','')
        headline = headline.replace('\,','')
        headline = headline.replace(' ','_')
        headline = headline.replace(':','_')
        headline = headline.replace('\\','_')
        headline = headline.replace('\/','_')
        headline = headline.replace('\x80','')
        headline = headline.replace('\x99','')
        headline = headline.replace('\x98','')
        headline = headline.replace('â','')
        headline = headline.replace(',','')
        headline = headline.replace('‘','')
        headline = headline.replace('/','_')
    print(headline)
    date_tag=speechtext.xpath("//span[@class='date']")
    date=date_tag[0].text
    date = date.replace(' ', '_')
    date = date.replace(',', '')
    #print(h2_tags.text)
    filename=str(index)+"_"+date+"_"+headline
    div_tag = speechtext.xpath("//div[@class='news-bg']")
    p_tag = speechtext.xpath("//p")
    speech=''
    for x in range(0,len(p_tag)):
        speech=speech+p_tag[x].text_content()
    f = open(filename+".txt",'w',encoding='utf-8')
    print(speech, file=f)
    index=index+1