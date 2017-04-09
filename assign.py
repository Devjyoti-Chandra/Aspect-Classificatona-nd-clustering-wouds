import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import ne_chunk, pos_tag
from nltk.stem import WordNetLemmatizer

import numpy as np
import pandas as pd
df=pd.read_table("aspect_annoated_file.txt", header=-1)

df.columns = ['Words','Aspect']
# print df['Aspect']

df['Bin']=df['Aspect'].map({'NASP':1,'ASP':0})    # Recoded categorical variable into numeric values

tag=pos_tag(df['Words'])    
t=[]

for i in range(len(df)):
	t.append(tag[i][1])			# extracted pos tags of each word

df['Tag']=t

lis=df.Tag.unique()				# Stores unique taggers in lis
q=range(0,(len(lis)))

h=[]
for i in range(len(df)):
    for j in range(len(lis)):
        if df['Tag'][i]==lis[j]:
            h.append(q[j])
            break

import gensim

# Word vectors of 40000 words loaded

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

word_vector=[[]]
asp=[[]]
word=[[]]
tagn=[]
lem=WordNetLemmatizer()
no_word=[]

for i in range(len(df)):
	if (df['Words'][i]).lower() in model.keys():
		word_vector.append(model[(df['Words'][i]).lower()])
		asp.append(df['Bin'][i])
		word.append((df['Words'][i]).lower())
		tagn.append(h[i])
	else :
		if 	lem.lemmatize((df['Words'][i]).lower()) in model.keys():			# If the words are not found then it is lemmatized and again checked
			lem_word = lem.lemmatize((df['Words'][i]).lower())
			word_vector.append(model[lem_word])
			asp.append(df['Bin'][i])
			word.append((df['Words'][i]).lower())
			tagn.append(h[i])
		else:
			no_word.append((df['Words'][i]).lower())							# list of words whose word vectors were not found

word_vector.pop(0)
asp.pop(0)
word.pop(0)	

# Calculating Max and Min value of word vector for normalizing tags

max_wv=max(map(lambda x: x[-1],word_vector))
min_wv=min(map(lambda x: x[-1],word_vector))

m=(len(tagn)-1)*min_wv/(min_wv-max_wv)     		# Tag element has been normalized between the max and min vector element
M=m-m/min_wv

tagn=pd.DataFrame(tagn)
tagn[0]=(tagn[0]-m)/(M-m)
tagn=tagn[0].tolist()

# for i in range(len(word_vector)):
# 	word_vector[i].append(tagn[i])

# print len(word_vector),len(asp),len(word)

ntrain=int(round(len(word_vector)*.75))
ntest=len(word_vector)-ntrain

train_x=pd.DataFrame(word_vector)[0:ntrain]
train_y=pd.DataFrame(asp)[0:ntrain]

test_x=pd.DataFrame(word_vector)[ntrain:]
test_y=pd.DataFrame(asp)[ntrain:]

train_tag=pd.DataFrame(tagn)[0:ntrain]
test_tag=pd.DataFrame(tagn)[ntrain:]

train_x=pd.concat([train_x,train_tag],axis=1)
test_x=pd.concat([test_x,test_tag],axis=1)

# SVM model

from sklearn import svm

model1 = svm.SVC(kernel='linear', C=1, gamma=1) 
model1.fit(train_x, train_y)
# print model1.score(train_x, train_y)			# Precision Score
pred=model1.predict(train_x)
pred_x=pd.DataFrame(pred)

#Predict Output
predicted= model1.predict(test_x)
pred_y=pd.DataFrame(predicted)

# Confusion Matrix
from sklearn.metrics import confusion_matrix

print "Model 1 (Window 1) : "
cnf_matrix = confusion_matrix(train_y, pred_x)
print "Train Data"
print cnf_matrix

from sklearn.metrics import precision_recall_fscore_support as score
precision, recall, fscore, support = score(train_y, pred_x)
print 'Precision: {}'.format(precision)
print 'Recall: {}'.format(recall)
print 'FScore: {}'.format(fscore)  

cnf_matrix = confusion_matrix(test_y, pred_y)
print "Test Data"
print cnf_matrix


precision, recall, fscore, support = score(test_y, pred_y)
print 'Precision: {}'.format(precision)
print 'Recall: {}'.format(recall)
print 'FScore: {}'.format(fscore)  

# Train Accuracy : 94.19 ||  Test Accuracy : 93.67

# Window is 3

word_vector3=[]


for i in range(len(word_vector)-1):
	if i==0:
		l=model['.']+word_vector[i]+word_vector[i+1]
	else :
		l=word_vector[i]+word_vector[i]+word_vector[i+1]
	word_vector3.append(l)

l=word_vector[len(word_vector)-2]+word_vector[len(word_vector)-1]+model['.']
word_vector3.append(l)

# print len(word_vector), len(word_vector3)

train1_x=pd.DataFrame(word_vector3)[0:ntrain]
test1_x=pd.DataFrame(word_vector3)[ntrain:]

train1_x=pd.concat([train1_x,train_tag],axis=1)
test1_x=pd.concat([test1_x,test_tag],axis=1)

model2 = svm.SVC(kernel='linear', C=1, gamma=1) 
model2.fit(train1_x, train_y)
# print model2.score(train1_x, train_y)
pred1=model2.predict(train1_x)
pred1_x=pd.DataFrame(pred1)

predicted1= model2.predict(test1_x)
pred1_y=pd.DataFrame(predicted1)

print "Model 2 (Window 3) : "
cnf_matrix1 = confusion_matrix(train_y, pred1_x)
print "Train Data"
print cnf_matrix1

precision, recall, fscore, support = score(train_y, pred1_x)
print 'Precision: {}'.format(precision)
print 'Recall: {}'.format(recall)
print 'FScore: {}'.format(fscore)  

cnf_matrix1 = confusion_matrix(test_y, pred1_y)
print "Test Data"
print cnf_matrix1

precision, recall, fscore, support = score(test_y, pred1_y)
print 'Precision: {}'.format(precision)
print 'Recall: {}'.format(recall)
print 'FScore: {}'.format(fscore) 

# Train Accuracy : 94.32  ||  Test Accuracy : 93.91

# Window is 5

word_vector5=[]


for i in range(len(word_vector)-2):
	if i==0:
		l=model['.']+model['.']+word_vector[i]+word_vector[i+1]+word_vector[i+2]
	elif i==1:
		l=model['.']+word_vector[i-1]+word_vector[i]+word_vector[i+1]+word_vector[i+2]	
	else :
		l=word_vector[i-2]+word_vector[i-1]+word_vector[i]+word_vector[i+1]+word_vector[i+2]
	word_vector5.append(l)

l=word_vector[len(word_vector)-4]+word_vector[len(word_vector)-3]+word_vector[len(word_vector)-2]+word_vector[len(word_vector)-1]+model['.']
word_vector5.append(l)
l=word_vector[len(word_vector)-3]+word_vector[len(word_vector)-2]+word_vector[len(word_vector)-1]+model['.']+model['.']
word_vector5.append(l)

# print len(word_vector), len(word_vector3), len(word_vector5)

train2_x=pd.DataFrame(word_vector5)[0:ntrain]
test2_x=pd.DataFrame(word_vector5)[ntrain:]

train2_x=pd.concat([train2_x,train_tag],axis=1)
test2_x=pd.concat([test2_x,test_tag],axis=1)

model3 = svm.SVC(kernel='linear', C=1, gamma=1) 
model3.fit(train2_x, train_y)
# print model3.score(train2_x, train_y)
pred2=model3.predict(train2_x)
pred2_x=pd.DataFrame(pred2)

predicted2= model3.predict(test2_x)
pred2_y=pd.DataFrame(predicted2)

print "Model 3 (Window 5) : "
cnf_matrix2 = confusion_matrix(train_y, pred2_x)
print "Train Data"
print cnf_matrix2

precision, recall, fscore, support = score(train_y, pred2_x)
print 'Precision: {}'.format(precision)
print 'Recall: {}'.format(recall)
print 'FScore: {}'.format(fscore)  

cnf_matrix2 = confusion_matrix(test_y, pred2_y)
print "Test Data"
print cnf_matrix2

precision, recall, fscore, support = score(test_y, pred2_y)
print 'Precision: {}'.format(precision)
print 'Recall: {}'.format(recall)
print 'FScore: {}'.format(fscore)

# Train Accuracy : 94.95  || Test Accuracy : 93.81





