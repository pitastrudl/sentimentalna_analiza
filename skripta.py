from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix

import numpy as np
import itertools
import matplotlib.pyplot as plt
import pandas
import sys


def tempwrite(stuff,filename):
	f = open(filename, "w")
	f.write(stuff)
	f.close()



#vse skupaj da v eno variablo
def get_all_data():

    with open(sys.argv[1], "r",encoding='UTF-8') as text_file: #podatki
        evaluation_data = text_file.read().split('\n')
         
    with open(sys.argv[2], "r",encoding='UTF-8') as text_file: # leksikon
        training_data = text_file.read().split('\n')
    

    return preprocessing_data(training_data), evaluation_data

#inserts into a list 
def preprocessing_data(data):
    processing_data = []
    for single_data in data:
	#we check that it splits into two parts and that the second part is not empty
        if len(single_data.split(",")) == 2 and single_data.split(",")[1] != "":
            processing_data.append(single_data.split(","))

    return processing_data


#here we learn everything 
def training_step(data, vectorizer):
    print(data)
    training_text = [data[0] for data in data] # stavke vrne vse, v stringu
    training_result = [data[1] for data in data] # nicle enke, sentiment, vse ksupaj
    training_text = vectorizer.fit_transform(training_text) # vrne neke vrednosti....cudne 
    #tempwrite(''.join(map(str,training_text)),"training_text")
    #tempwrite(''.join(map(str,training_result)),"training_result")
    return BernoulliNB().fit(training_text, training_result)

#analyses each line
def analyse_text(classifier, vectorizer, text):
    return text, classifier.predict(vectorizer.transform([text]))

def simple_evaluation(evaluation_data):
    evaluation_text = evaluation_data
    total = len(evaluation_text)
    for index in range(0, total):
        analysis_result = analyse_text(classifier, vectorizer, evaluation_text[index])
        text, result = analysis_result
        print(text,result)
    
        

#--------------------------------------------------------------------------------------------------

training_data, evaluation_data = get_all_data()
#tempwrite(''.join(map(str,training_data)),"training_data")
#tempwrite(''.join(map(str,evaluation_data)),"evaluation_data")
vectorizer = CountVectorizer(binary = 'true') # naredimo classifer
classifier = training_step(training_data, vectorizer) # se nauci 
simple_evaluation(evaluation_data)

def asdasda():


    print()
    result = classifier.predict(vectorizer.transform(["I love this movie!"])) # predicting
    print(simple_evaluation(evaluation_data))


