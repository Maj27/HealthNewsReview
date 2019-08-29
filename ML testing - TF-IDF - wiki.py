



# -*- coding: utf-8 -*-
"""
Created on  May 1 18:58:00 2018

@author: maj27
This program reads a text corpus (pos neg folders containing files of text format)
Then learn different classifiers and test them on a separate test set

IMPORTANT note:
  to speed up the process of NER I did it separeately using preprocessingNER file,
  it does all the converstion to NER then save to a dump file as:
  
  1- When training and validating I used data.p
   here the file has training and validation sets
  2- When final testing I used data_2.p
   here the file has combined training and validation sets as train+val
   then tesing on testing set
   
   for 1 (validation)  use:
       preprocessed = 'C:/Users/'+machinename+'/New folder/Dropbox/PhD Brighton/Dataset/healthnewsreview_org/Well done 5 and 10 inverted/Classified Story/Criteria '+dataset+'/data.p'
   for 2 (testing) use:
       preprocessed = 'C:/Users/'+machinename+'/New folder/Dropbox/PhD Brighton/Dataset/healthnewsreview_org/Well done 5 and 10 inverted/Classified Story/Criteria '+dataset+'/data_2.p'
   


"""

import collections

from nltk.corpus import CategorizedPlaintextCorpusReader

import random
from nltk.stem.snowball import SnowballStemmer
import numpy as np
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
 
from nltk.classify import maxent
from sklearn import tree

from scipy import stats
from collections import Counter
from sklearn import preprocessing
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn import cross_validation
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import cross_val_predict
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import pickle

############################################################################### 
import matplotlib.pyplot as plt


#def plot_coefficients(classifier, feature_names, top_features=40):
#    coef = classifier.coef_.ravel()
#    top_negative_coefficients = np.argsort(coef)[-top_features:]
#    top_coefficients = np.hstack([top_negative_coefficients])
#    # create plot
#    plt.figure(figsize=(15, 5))
#    colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
#    plt.bar(np.arange( top_features), coef[top_coefficients], color=colors)
#    feature_names = np.array(feature_names)
#    plt.xticks(np.arange(1,  1 + top_features), feature_names[top_coefficients], rotation=60, ha='right')
#    plt.show()
#    
#    topfeatures = feature_names[top_coefficients]  
#    toplist = topfeatures.tolist()
#    toplist.reverse()
#  
#    for feature in toplist: 
#        with open('c:/features.txt', 'a') as the_file:
#            the_file.write(feature + ', ')
#   

def plot_coefficients(classifier, feature_names, top_features=40):
     coef = classifier.coef_.ravel()
     top_positive_coefficients = np.argsort(coef)[-top_features:]
     top_negative_coefficients = np.argsort(coef)[:top_features]
     top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
     # create plot
     plt.figure(figsize=(15, 5))
     colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
     plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
     feature_names = np.array(feature_names)
     plt.xticks(np.arange(1,  1 + top_features), feature_names[top_coefficients], rotation=60, ha='right')
     plt.show()
     
     topfeatures = feature_names[top_coefficients]  
     toplist = topfeatures.tolist()
     toplist.reverse()
  
     for feature in toplist: 
         with open('c:/features.txt', 'a') as the_file:
             the_file.write(feature + ', ')
             

 ###############################################################################
               
def get_evaluation_metrics(reference, result):

    evaluation_metrics = collections.namedtuple('evaluation_metrics', ['accuracy', 'avg_recall','avg_precision','avg_F1']) 
    accuracy = accuracy_score(reference,result)  
    avg_recall = recall_score(reference,result,average='weighted')    
    avg_precision = precision_score(reference,result,average='weighted')    
    avg_F1 = f1_score(reference,result,average='weighted')  
   
    evm = evaluation_metrics(accuracy,avg_recall,avg_precision,avg_F1)
    return evm

###############################################################################
                 ###############################################################################
def read_words(words_file):
    with open(words_file, 'r') as f:
        words_list = []
        for word in f:
            words_list.append(word.rstrip())
        return words_list
###############################################################################
def normalize_text_wiki(text, lemmatize = None,remove_stop = None):
    
    import re
    text = re.compile(r'\W+', re.UNICODE).split(text)
    comparitive_forms = read_words('C:/Users/maj27/Downloads/wiktionarylists/wiktionarylists/comparative_forms.txt')
    modal_adverbs = read_words('C:/Users/maj27/Downloads/wiktionarylists/wiktionarylists/modal_adverbs.txt')
    act_adverbs = read_words('C:/Users/maj27/Downloads/wiktionarylists/wiktionarylists/act_adverbs.txt')
    manner_adverbs = read_words('C:/Users/maj27/Downloads/wiktionarylists/wiktionarylists/manner_adverbs.txt')
    superlative_forms = read_words('C:/Users/maj27/Downloads/wiktionarylists/wiktionarylists/superlative_forms.txt')
    numerals = read_words('C:/Users/maj27/Downloads/wiktionarylists/wiktionarylists/numerals.txt')
    degree_adverbs = read_words('C:/Users/maj27/Downloads/wiktionarylists/wiktionarylists/degree_adverbs.txt')
    auxiliary_verbs = read_words('C:/Users/maj27/Downloads/wiktionarylists/wiktionarylists/auxiliary_verbs.txt')
    
    wiki_text = []  
    for word in text:
        if word in comparitive_forms:
            wiki_text.append('COMPARITIVE')
#        elif word in numerals:
#            wiki_text.append('NUMERAL')
##        elif word in degree_adverbs:
##            wiki_text.append('DEGREE_ADVERB')
#        elif word in auxiliary_verbs:
#            wiki_text.append('AUXILIARY_VERB')
         #########           
#        elif word in modal_adverbs:
#            wiki_text.append('MODAL')
#        elif word in act_adverbs:
#            wiki_text.append('ACT_ADVERBS')
#        elif word in manner_adverbs:
#            wiki_text.append('MANNER_VERBS')
#        elif word in superlative_forms:
#            wiki_text.append('SUPERLATIVE')
                
        
        else:
            wiki_text.append(word)
      
    " ".join([ word for word in wiki_text])      
    return wiki_text                         


##################################################################################        
############################### Start Execution Point #############################        
###################################################################################        

from time import gmtime, strftime
print strftime("%Y-%m-%d %H:%M:%S", gmtime())
# Uni
machinename = 'maj27'

# Laptop
#machinename = 'majed'
# all data

import xlsxwriter


# Create an new Excel file and add a worksheet.
workbook = xlsxwriter.Workbook('c:\demoTestResults.xlsx')
sheet = workbook.add_worksheet('resu')
#import xlwt
#
#book = xlwt.Workbook()
#sheet = book.add_sheet('Cross val results')

j=0
for i in range(6,7):
    dataset = str(i+1)
  
    preprocessed = 'C:/Users/'+machinename+'/New folder/Dropbox/PhD Brighton/Dataset/healthnewsreview_org/Well done 5 and 10 inverted/Classified Story/Criteria '+dataset+'/data.p'
   

    print 'Testing, Criteria ' + dataset + ' :' 
    
    
  # Getting back the objects:
    with open(preprocessed) as f:  # Python 3: open(..., 'rb')
        train_data, train_labels, test_data,test_labels,test_fileIds = pickle.load(f)
    
    train_data = [' '.join(normalize_text_wiki(document)) for document in train_data]
    test_data = [' '.join(normalize_text_wiki(document)) for document in test_data]
    
    
    
    #cv1 = CountVectorizer(ngram_range=(1, 3),stop_words = 'english',max_features  = 3000)
    cv1 = TfidfVectorizer(ngram_range=(1, 3), lowercase=False, max_features  = 7000, max_df = 0.7 , min_df=5, sublinear_tf  =True)
    #cv1 = TfidfVectorizer(ngram_range=(1, 3),min_df = 0.2)
    
    train_vectors = cv1.fit_transform(train_data)
    test_vectors = cv1.transform(test_data)
    features_names = cv1.get_feature_names()
    
    ########################################################################
    ######   apply feature selection
#    from sklearn.feature_selection import SelectFromModel
#    
#    clf_selector = LogisticRegression() 
#    clf_selector.fit(train_vectors, train_labels)
#    model = SelectFromModel(clf_selector, prefit=True,threshold=0.2)
#    train_reduced_features = model.transform(train_vectors)
#        
#    test_reduced_features = model.transform(test_vectors)
#    
    #########################################################################
    
    feature_cut = 200
    select = SelectKBest(chi2, k=feature_cut)
    train_reduced_features = select.fit_transform(train_vectors, train_labels)
    
    print train_reduced_features.shape
    
    test_reduced_features = select.transform(test_vectors)
    
    
    #########################################################################
    print 'Linear SVM classifier'
    
    
    SVMclassifier = LinearSVC( C=8)
    SVMclassifier.fit(train_reduced_features, train_labels) 
    predictions = SVMclassifier.predict(test_reduced_features)
    evaluations =  get_evaluation_metrics(test_labels, predictions)
    print '          {}        {}        {}  '.format(round(evaluations[1]*100,2), round(evaluations[2]*100,2),round(evaluations[3]*100,2))
    
    sheet.write(j,0,'Criteria' + dataset + '')
    sheet.write(j+1,0,'classifier 1')
    sheet.write(j+1,1,round(evaluations[1]*100,2))
    sheet.write(j+1,2,round(evaluations[2]*100,2))
    sheet.write(j+1,3,round(evaluations[3]*100,2))
    
    #print 'MLP Classifier '
    #
    #MLP = MLPClassifier()
    #MLP.fit(train_reduced_features, train_labels) 
    #predictions = MLP.predict(test_reduced_features)
    #evaluations =  get_evaluation_metrics(test_labels, predictions)
    #print '          {}        {}        {}  '.format(round(evaluations[1]*100,2), round(evaluations[2]*100,2),round(evaluations[3]*100,2))
    #
    #
    #
    #print 'Decision Tree Classifier' 
    #
    #clf = tree.DecisionTreeClassifier()
    #clf.fit(train_reduced_features, train_labels) 
    #predictions = clf.predict(test_reduced_features)
    #evaluations =  get_evaluation_metrics(test_labels, predictions)
    #print '          {}        {}        {}  '.format(round(evaluations[1]*100,2), round(evaluations[2]*100,2),round(evaluations[3]*100,2))
    
    
    ##########################  
#    print 'Logistic Regression Classifier'
#    LR = LogisticRegression()
#    LR.fit(train_reduced_features, train_labels) 
#    predictions = LR.predict(test_reduced_features)
#    evaluations =  get_evaluation_metrics(test_labels, predictions)
#    print '          {}        {}        {}  '.format(round(evaluations[1]*100,2), round(evaluations[2]*100,2),round(evaluations[3]*100,2))
    
#    sheet.write(j+2,0,'classifier 2')
#    sheet.write(j+2,1,round(evaluations[1]*100,2))
#    sheet.write(j+2,2,round(evaluations[2]*100,2))
#    sheet.write(j+2,3,round(evaluations[3]*100,2)) 
#    
#      ###########################  
#    print 'MultinomialNB Classifier'
#    NB = MultinomialNB()
#    NB.fit(train_reduced_features, train_labels) 
#    predictions = NB.predict(test_reduced_features)
#    evaluations =  get_evaluation_metrics(test_labels, predictions)
#    print '          {}        {}        {}  '.format(round(evaluations[1]*100,2), round(evaluations[2]*100,2),round(evaluations[3]*100,2))
#    
#    sheet.write(j+3,0,'classifier 3')
#    sheet.write(j+3,1,round(evaluations[1]*100,2))
#    sheet.write(j+3,2,round(evaluations[2]*100,2))
#    sheet.write(j+3,3,round(evaluations[3]*100,2)) 
#    ###########################  
          
    print 'Random Forest Classifier'   
    
    from sklearn.ensemble import RandomForestClassifier
    clf2 = RandomForestClassifier(n_estimators=50)
    clf2.fit(train_reduced_features, train_labels) 
    predictions = clf2.predict(test_reduced_features)
    evaluations =  get_evaluation_metrics(test_labels, predictions)
    print '          {}        {}        {}  '.format(round(evaluations[1]*100,2), round(evaluations[2]*100,2),round(evaluations[3]*100,2))
    
    sheet.write(j+2,0,'classifier 2')
    sheet.write(j+2,1,round(evaluations[1]*100,2))
    sheet.write(j+2,2,round(evaluations[2]*100,2))
    sheet.write(j+2,3,round(evaluations[3]*100,2))   
    ######################################################################
    # Find misclassified docsuments
    test_labels = np.asarray(test_labels)
    misclassified_indices = np.where(test_labels != clf2.predict(test_reduced_features))
    test_fileIds_ = np.array(test_fileIds)
    misclassified_docs = test_fileIds_[misclassified_indices[0]]
    
    print misclassified_docs
    
    
    ###################################################################################
    
    # Print the reduced features names (only works with selectBestK )
    select = SelectKBest(chi2, k=feature_cut )
    features_names = cv1.get_feature_names()
    train_reduced_features = select.fit(train_vectors, train_labels)
    idxs_selected = train_reduced_features.get_support(indices=True)
    reduced_features_names = [features_names[i] for i in idxs_selected]
    
    print reduced_features_names[0:100]
    
    
    with open('c:/features.txt', 'a') as the_file:
        the_file.write('\nTesting, Criteria ' + dataset + ' :\n')
        

    
    j=j+3
    #plot_coefficients(SVMclassifier, reduced_features_names, top_features=50)
    

    
workbook.close()    
print strftime("%Y-%m-%d %H:%M:%S", gmtime())