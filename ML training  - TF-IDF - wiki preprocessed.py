# -*- coding: utf-8 -*-
"""
Created on  May 1 18:58:00 2018
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
from sklearn.metrics import confusion_matrix 

from nltk.classify import maxent

from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn import cross_validation
from sklearn import tree
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import cross_val_predict
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import pickle
############################################################################### 
import matplotlib.pyplot as plt


def plot_coefficients(classifier, feature_names, top_features=40):
 coef = classifier.coef_.ravel()
 top_negative_coefficients = np.argsort(coef)[-top_features:]
 top_coefficients = np.hstack([top_negative_coefficients])
 # create plot
 plt.figure(figsize=(15, 5))
 colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
 plt.bar(np.arange( top_features), coef[top_coefficients], color=colors)
 feature_names = np.array(feature_names)
 plt.xticks(np.arange(1,  1 + top_features), feature_names[top_coefficients], rotation=60, ha='right')
 plt.show()
 
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
    comparitive_forms = read_words('C:/Users/maj27/Downloads/wiktionarylists/wiktionarylists/lemmatised/comparative_forms.txt')
    modal_adverbs = read_words('C:/Users/maj27/Downloads/wiktionarylists/wiktionarylists/lemmatised/modal_adverbs.txt')
    act_adverbs = read_words('C:/Users/maj27/Downloads/wiktionarylists/wiktionarylists/lemmatised/act_adverbs.txt')
    manner_adverbs = read_words('C:/Users/maj27/Downloads/wiktionarylists/wiktionarylists/lemmatised/manner_adverbs.txt')
    superlative_forms = read_words('C:/Users/maj27/Downloads/wiktionarylists/wiktionarylists/lemmatised/superlative_forms.txt')
    numerals = read_words('C:/Users/maj27/Downloads/wiktionarylists/wiktionarylists/lemmatised/numerals.txt')
    degree_adverbs = read_words('C:/Users/maj27/Downloads/wiktionarylists/wiktionarylists/lemmatised/degree_adverbs.txt')
    auxiliary_verbs = read_words('C:/Users/maj27/Downloads/wiktionarylists/wiktionarylists/lemmatised/auxiliary_verbs.txt')

    wiki_text = []  
    for word in text:
#        if word in comparitive_forms:
#            wiki_text.append('COMPARITIVE')
        if word in numerals:
            wiki_text.append('NUMERAL')
#        elif word in degree_adverbs:
#            wiki_text.append('DEGREE_ADVERB')
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


###################################################################################        
############################### Start Execution Point #############################        
###################################################################################        
from time import gmtime, strftime
print strftime("%Y-%m-%d %H:%M:%S", gmtime())

# Uni
machinename = 'maj27'

# Laptop
#machinename = 'majed'
# all data

#mydir = 'C:/Users/'+machinename+'/New folder/Dropbox/PhD Brighton/Dataset/healthnewsreview_org/Classified News/Training'
#mydir = 'C:/Users/'+machinename+'/New folder/Dropbox/PhD Brighton/Dataset/healthnewsreview_org/NA is ignored/Classified Story/Criteria 3'

import xlsxwriter


# Create an new Excel file and add a worksheet.
workbook = xlsxwriter.Workbook('c:\demoTrainResults.xlsx')
sheet = workbook.add_worksheet('resu')
#import xlwt
#
#book = xlwt.Workbook()
#sheet = book.add_sheet('Cross val results')

j=0
for i in range(1):
    
    dataset = str(i+1)

    preprocessed = 'C:/Users/'+machinename+'/New folder/Dropbox/PhD Brighton/Dataset/healthnewsreview_org/Well done 5 and 10 inverted/Classified Story/Criteria '+dataset+'/data.p'
   

    print 'Training, Criteria ' + dataset + ' :' 
    
    
  # Getting back the objects:
    with open(preprocessed) as f:  # Python 3: open(..., 'rb')
        train_data, train_labels, test_data,test_labels,test_fileIds = pickle.load(f)
    
    train_data = [' '.join(normalize_text_wiki(document)) for document in train_data]

    
    
    
    #cv1 = CountVectorizer(ngram_range=(1, 3),stop_words = 'english',max_features  = 3000)
    cv1 = TfidfVectorizer(ngram_range=(1, 3), max_features  = 7000, lowercase=False, max_df = 0.7 , min_df=5, sublinear_tf  =True)
    #cv1 = TfidfVectorizer(ngram_range=(1, 3),min_df = 0.2)
    
    #train_vectors = cv1.fit_transform(train_data)
    #features_names = cv1.get_feature_names()
    
    ########################################################
    
    scores1 = []
    scores2 = []
    scores3 = []
    scores4 = []
    scores5 = []
    
 
    kf = KFold(n_splits=5, shuffle=True, random_state=5)
    
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    
    
    for i, (train_index, test_index) in enumerate(kf.split(train_data, train_labels)): 
        #print("TRAIN:", train_index, "TEST:", test_index)
        xtrain, xtest = train_data[train_index], train_data[test_index]
        ytrain, ytest = train_labels[train_index], train_labels[test_index]   
        
        
   
    
        xtrain = cv1.fit_transform(xtrain) 
        xtest = cv1.transform(xtest)
     ########################################################################   
     # feature selection using SelectKBest
        feature_cut = 400
        select = SelectKBest(chi2, k=feature_cut)
        train_reduced_features = select.fit_transform(xtrain, ytrain)   
        test_reduced_features = select.transform(xtest)
     #######################################################################
     ## feature selection using SelectFromModel
        
#        clf_selector = LogisticRegression()
#        from sklearn.feature_selection import SelectFromModel  
#        clf_selector.fit(xtrain, ytrain)
#        model = SelectFromModel(clf_selector, prefit=True,threshold=0.2)
#        train_reduced_features = model.transform(xtrain)
#        
#        test_reduced_features = model.transform(xtest)
    #
    
     ## evaluation   
    ## classify using svm
        SVMclassifier = LinearSVC(C=8)
        
        SVMclassifier.fit(train_reduced_features, ytrain) 
        predictions = SVMclassifier.predict(test_reduced_features)
        score =  get_evaluation_metrics(ytest, predictions)
        scores1.append([score.accuracy, score.avg_recall, score.avg_precision,score.avg_F1])
    
    #    MLP = MLPClassifier()
    #    
    #    MLP.fit(train_reduced_features, ytrain) 
    #    predictions = MLP.predict(test_reduced_features)
    #    score =  get_evaluation_metrics(ytest, predictions)
    #    scores2.append([score.accuracy, score.avg_recall, score.avg_precision,score.avg_F1])
    #
    #    clf = tree.DecisionTreeClassifier()
    #    clf.fit(train_reduced_features, ytrain) 
    #    predictions = clf.predict(test_reduced_features)
    #    score =  get_evaluation_metrics(ytest, predictions)
    #    scores3.append([score.accuracy, score.avg_recall, score.avg_precision,score.avg_F1])

        # classify using LogisticRegression()

#      ###########################  
#        LR = LogisticRegression()
#        LR.fit(train_reduced_features, ytrain) 
#        predictions = LR.predict(test_reduced_features)
#        score =  get_evaluation_metrics(ytest, predictions)
#        scores2.append([score.accuracy, score.avg_recall, score.avg_precision,score.avg_F1])
#       ###########################     
#        NB = MultinomialNB()
#        NB.fit(train_reduced_features, ytrain) 
#        predictions = NB.predict(test_reduced_features)
#        score =  get_evaluation_metrics(ytest, predictions)
#        scores3.append([score.accuracy, score.avg_recall, score.avg_precision,score.avg_F1])
#    #
        ###########################  
        from sklearn.ensemble import RandomForestClassifier
        clf2 = RandomForestClassifier(n_estimators=50)
        clf2.fit(train_reduced_features, ytrain) 
        predictions = clf2.predict(test_reduced_features)
        score =  get_evaluation_metrics(ytest, predictions)
        scores4.append([score.accuracy, score.avg_recall, score.avg_precision,score.avg_F1])
        
    

    
    
    
   ###########################    
    tran_scores = np.array(scores1)
    evaluations = tran_scores.mean(axis=0)
    
    print 'Linear SVM classifier'
    print '          {}        {}        {}  '.format(round(evaluations[1]*100,2), round(evaluations[2]*100,2),round(evaluations[3]*100,2))
    
    sheet.write(j,0,'Criteria' + dataset + '')
    sheet.write(j+1,0,'classifier 1')
    sheet.write(j+1,1,round(evaluations[1]*100,2))
    sheet.write(j+1,2,round(evaluations[2]*100,2))
    sheet.write(j+1,3,round(evaluations[3]*100,2))
    
  ###########################     
#    tran_scores = np.array(scores2)
#    evaluations = tran_scores.mean(axis=0)
#    
#    print 'Logistic Regression classifier'
#    print '          {}        {}        {}  '.format(round(evaluations[1]*100,2), round(evaluations[2]*100,2),round(evaluations[3]*100,2))
#    
#    sheet.write(j+2,0,'classifier 2')
#    sheet.write(j+2,1,round(evaluations[1]*100,2))
#    sheet.write(j+2,2,round(evaluations[2]*100,2))
#    sheet.write(j+2,3,round(evaluations[3]*100,2))   
#
#  ###########################     
#    tran_scores = np.array(scores3)
#    evaluations = tran_scores.mean(axis=0)
#    
#    print 'Naive bayes classifier'
#    print '          {}        {}        {}  '.format(round(evaluations[1]*100,2), round(evaluations[2]*100,2),round(evaluations[3]*100,2))
#    
#    sheet.write(j+3,0,'classifier 3')
#    sheet.write(j+3,1,round(evaluations[1]*100,2))
#    sheet.write(j+3,2,round(evaluations[2]*100,2))
#    sheet.write(j+3,3,round(evaluations[3]*100,2))    
#       
#    
  ###########################    
    tran_scores = np.array(scores4)
    evaluations = tran_scores.mean(axis=0)
    
    print 'Random Forest Classifier'
    print '          {}        {}        {}  '.format(round(evaluations[1]*100,2), round(evaluations[2]*100,2),round(evaluations[3]*100,2))
    
    sheet.write(j+2,0,'classifier 2')
    sheet.write(j+2,1,round(evaluations[1]*100,2))
    sheet.write(j+2,2,round(evaluations[2]*100,2))
    sheet.write(j+2,3,round(evaluations[3]*100,2))     
    
    
    j=j+3
    ########################### 
    # Print the reduced features names (only works with selectBestK )
    select = SelectKBest(chi2, k=feature_cut)
    features_names = cv1.get_feature_names()
    train_reduced_features = select.fit(xtrain, ytrain)
    idxs_selected = train_reduced_features.get_support(indices=True)
    reduced_features_names = [features_names[i] for i in idxs_selected]
    
    print reduced_features_names[0:20]
    
    
    
    
    
    
    #plot_coefficients(SVMclassifier, features_names, top_features=40)
    
    
    
    
    
    
    
    ###### visualize a decision tree
    #import graphviz 
    #labs = ['pos' if i==1 else 'neg' for i in labels]
    #dot_data = tree.export_graphviz(clf2, out_file=None, 
    #                         feature_names=features_names,  
    #                         class_names=labs,  
    #                         filled=True, rounded=True,  
    #                         special_characters=True)
    #
    #dot_data = tree.export_graphviz(clf2, out_file=None) 
    #graph = graphviz.Source(dot_data) 
    #graph.render("Cri") 
    
    
    #############################################
    
    


workbook.close()    
print strftime("%Y-%m-%d %H:%M:%S", gmtime())

#book.save("c:\results_excel.xls")    
    