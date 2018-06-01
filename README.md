# BBC WorkSpace

### News Analysis

The following folder contains analysis on the PMs Speeches and also on a Real-time FactChecker to verify truth of given sentence or fact in real time, whose UI can be developed further, while the algorithm can be made more robust. 

#### PMs Speeches

Classifies PMs Speeches into various subject using the K means algorithm, using the Count Vectorizer along with the TFidf vectorizer. The number of subjects to which it can be classified can be changed. 


#### Real-time FactChecker

The following program does real time fact checking on a document. To be further developed so that it can be used as a tool for journalists. 

At the heart of this is A highly modified TextRank Algorithm which resembles that of the pageRank algorithm, however has a number of individual changes and variables. The number of sources you wish the fact checker should check through can be modified. 

The following program is optimized by applying a map reduce function wherever possible, to deliver results quicker. Since its main source is the internet, an initial authencity of the information being put on the website must be checked, but since we compare results from multiple websites making sure that content of each of them are not entirely same, we can avoid such a function, thus saving time and still maintinaing the same accuracy. 


