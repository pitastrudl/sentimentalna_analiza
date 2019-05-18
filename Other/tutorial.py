from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix

import numpy as np
import itertools
import matplotlib.pyplot as plt
import pandas


def tempwrite(stuff,filename):
	f = open(filename, "w")
	f.write(stuff)
	f.close()

def preprocessing_step():
    data = get_all_data()
    processing_data = preprocessing_data(data)

    return split_data(processing_data)

#vse skupaj da v eno variablo
def get_all_data():
    root = "Data/"

    with open(root + "imdb_labelled.txt", "r") as text_file:
        data = text_file.read().split('\n')
         
    with open(root + "amazon_cells_labelled.txt", "r") as text_file:
        data += text_file.read().split('\n')

    with open(root + "yelp_labelled.txt", "r") as text_file:
        data += text_file.read().split('\n')

    return data

#inserts into a list 
def preprocessing_data(data):
    processing_data = []
    for single_data in data:
	#we check that it splits into two parts and that the second part is not empty
        if len(single_data.split("\t")) == 2 and single_data.split("\t")[1] != "":
            processing_data.append(single_data.split("\t"))

    return processing_data



#split the data for training and evaluation
def split_data(data):
    total = len(data)
    training_ratio = 0.75
    training_data = []
    evaluation_data = []

    for indice in range(0, total):
        if indice < total * training_ratio:
            training_data.append(data[indice])
        else:
            evaluation_data.append(data[indice])

    return training_data, evaluation_data

#here we learn everything 
def training_step(data, vectorizer):

    training_text = [data[0] for data in data] # stavke vrne vse, v stringu
    training_result = [data[1] for data in data] # nicle enke, sentiment, vse ksupaj
    training_text = vectorizer.fit_transform(training_text) # vrne neke vrednosti....cudne 
    #tempwrite(''.join(map(str,training_text)),"training_text")
    #tempwrite(''.join(map(str,training_result)),"training_result")
    return BernoulliNB().fit(training_text, training_result)

#analizira vsako linijo
def analyse_text(classifier, vectorizer, text):
    return text, classifier.predict(vectorizer.transform([text]))

#sprinta kar vrne analyse_text
def print_result(result):
    text, analysis_result = result
    print_text = "Positive" if analysis_result[0] == '1' else "Negative"
    print(text, ":", print_text)


def simple_evaluation(evaluation_data):
    evaluation_text     = [evaluation_data[0] for evaluation_data in evaluation_data] # besedila
    evaluation_result   = [evaluation_data[1] for evaluation_data in evaluation_data] #sentimenti
    #tempwrite(''.join(map(str,evaluation_text)),"evaluation_text")
    #tempwrite(''.join(map(str,evaluation_result)),"evaluation_result")
    total = len(evaluation_text)
    corrects = 0
    for index in range(0, total):
        analysis_result = analyse_text(classifier, vectorizer, evaluation_text[index])
        text, result = analysis_result
        corrects += 1 if result[0] == evaluation_result[index] else 0 #tole gleda kok je točno

    return corrects * 100 / total

def create_confusion_matrix(evaluation_data):
    evaluation_text     = [evaluation_data[0] for evaluation_data in evaluation_data]
    actual_result       = [evaluation_data[1] for evaluation_data in evaluation_data]
    prediction_result   = []
    for text in evaluation_text:
        analysis_result = analyse_text(classifier, vectorizer, text)
        prediction_result.append(analysis_result[1][0])
    
    matrix = confusion_matrix(actual_result, prediction_result)
    return matrix


#--------------------------------------------------------------------------------------------------
training_data, evaluation_data = preprocessing_step()
tempwrite(''.join(map(str,training_data)),"training_data")
tempwrite(''.join(map(str,evaluation_data)),"evaluation_data")

vectorizer = CountVectorizer(binary = 'true') # naredimo classifer
classifier = training_step(training_data, vectorizer) # se nauci 
print(simple_evaluation(evaluation_data))
def asdasda():

	
	
	result = classifier.predict(vectorizer.transform(["I love this movie!"])) # predicting
	print(simple_evaluation(evaluation_data))
	confusion_matrix_result= create_confusion_matrix(evaluation_data)

	# visualisation-------------------------------
	test = pandas.DataFrame(
		confusion_matrix_result, 
		columns=["Negatives", "Positives"],
		index=["Negatives", "Positives"])
	print(test)
	classes = ["Negatives", "Positives"]

	plt.figure()
	plt.imshow(confusion_matrix_result, interpolation='nearest', cmap=plt.cm.Blues)
	plt.title("Confusion Matrix - Sentiment Analysis")
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	text_format = 'd'
	thresh = confusion_matrix_result.max() / 2.
	for row, column in itertools.product(range(confusion_matrix_result.shape[0]), range(confusion_matrix_result.shape[1])):
	    plt.text(column, row, format(confusion_matrix_result[row, column], text_format),
		     horizontalalignment="center",
		     color="white" if confusion_matrix_result[row, column] > thresh else "black")

	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.tight_layout()

	plt.show()

