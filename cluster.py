import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import ne_chunk, pos_tag
from nltk.stem import WordNetLemmatizer

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans

# Type the words in the list sen you want to cluster
sen=['food', 'Biriyani', 'fish', 'song', 'staff', 'waiting', 'theme', 'service']
# Type the number of clusters you want
n=4

df=pd.DataFrame(sen)

import gensim

model = {}
def loadGloveModel(gloveFile):
    print "Loading Glove Model"
    f = open(gloveFile,'r')
    
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = [float(val) for val in splitLine[1:]]
        model[word] = embedding
    print "Done.",len(model)," words loaded!"
    return model		

loadGloveModel('glove.6B.50d.txt')

df.columns = ['Words']

tag=pos_tag(df['Words'])
t=[]

for i in range(len(df)):
	t.append(tag[i][1])

df['Tag']=t

lis=df.Tag.unique()
q=range(0,(len(lis)))

h=[]
for i in range(len(df)):
    for j in range(len(lis)):
        if df['Tag'][i]==lis[j]:
            h.append(q[j])
            break


word_vector=[]
word=[]
tagn=[]
lem=WordNetLemmatizer()
no_word=[]

for i in range(len(df)):
	if (df['Words'][i]).lower() in model.keys():
		word_vector.append(model[(df['Words'][i]).lower()])
		word.append((df['Words'][i]).lower())
		tagn.append(h[i])
	else :
		if 	lem.lemmatize((df['Words'][i]).lower()) in model.keys():			# If the words are not found then it is lemmatized and again checked
			lem_word = lem.lemmatize((df['Words'][i]).lower())
			word_vector.append(model[lem_word])
			word.append((df['Words'][i]).lower())
			tagn.append(h[i])
		else:
			no_word.append((df['Words'][i]).lower())							# list of words whose word vectors were not found

x=pd.DataFrame(word_vector)

kmeans = KMeans(n_clusters=n, random_state=0).fit(x)

# print kmeans.labels_
# print word

x[50]=kmeans.labels_

cluster=[]
final=[]
for i in range(0, n):
	for j in range(len(x)):
		if i==x[50][j]:
			cluster.append(word[j])
	if len(cluster)>0 :
		final.append(cluster)
	cluster=[]			

print final

d_intra=0
temp=0
for i in range(len(x)):
	for j in range(i+1,len(x)):
		if (x[50][i])==(x[50][j]):
			for k in range(len(x.columns)-1):
				temp=temp+(x[k][i]-x[k][j])**2 
			d_intra=d_intra+temp**(1/2.0)
			temp=0	

# Sum of distances between word vectors of each cluster

d_intra=0
temp=0
for i in range(len(x)):
	for j in range(i+1,len(x)):
		if (x[50][i])==(x[50][j]):
			for k in range(len(x.columns)-1):
				temp=temp+(x[k][i]-x[k][j])**2 
			d_intra=d_intra+temp**(1/2.0)
			temp=0

# Sum of distances between word vectors of one cluster to word vectors of other clusters

d_inter=0
temp=0
for i in range(len(x)):
	for j in range(i+1,len(x)):
		if (x[50][i])!=(x[50][j]):
			for k in range(len(x.columns)-1):
				temp=temp+(x[k][i]-x[k][j])**2 
			d_inter=d_inter+temp**(1/2.0)
			temp=0
