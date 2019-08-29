# -*- coding: utf-8 -*-
"""
Created on  Nov 1 18:58:00 2017

@author: maj27
This program reads a text corpus (pos neg folders containing files of text format)
Then learn different classifiers and test them on a separate test set


IMPORTANT note:
  to speed up the process of NER this file does all the converstion to NER then save to a dump file as:
  
  1- When training and validating I used data.p
   here the file has training and validation sets
  2- When final testing I used data_2.p
   here the file has combined training and validation sets as train+val
   then tesing on testing set
   
   for 1 (validation)  use:
        
    train_dir = 'C:/Users/'+machinename+'/New folder/Dropbox/PhD Brighton/Dataset/healthnewsreview_org/Well done 5 and 10 inverted/Classified Story/Criteria '+dataset+'/Training'
    test_dir = 'C:/Users/'+machinename+'/New folder/Dropbox/PhD Brighton/Dataset/healthnewsreview_org/Well done 5 and 10 inverted/Classified Story/Criteria '+dataset+'/validation'

    preprocessed = 'C:/Users/'+machinename+'/New folder/Dropbox/PhD Brighton/Dataset/healthnewsreview_org/Well done 5 and 10 inverted/Classified Story/Criteria '+dataset+'/data.p'
  
   for 2 (testing) use:
      train_dir = 'C:/Users/'+machinename+'/New folder/Dropbox/PhD Brighton/Dataset/healthnewsreview_org/Well done 5 and 10 inverted/Classified Story/Criteria '+dataset+'/Train+val'
      test_dir = 'C:/Users/'+machinename+'/New folder/Dropbox/PhD Brighton/Dataset/healthnewsreview_org/Well done 5 and 10 inverted/Classified Story/Criteria '+dataset+'/Testing'

      preprocessed = 'C:/Users/'+machinename+'/New folder/Dropbox/PhD Brighton/Dataset/healthnewsreview_org/Well done 5 and 10 inverted/Classified Story/Criteria '+dataset+'/data_2.p'
  


"""


from nltk.corpus import CategorizedPlaintextCorpusReader

import pickle
###############################################################################
from nltk.tag import StanfordNERTagger
st = StanfordNERTagger('C:/stanford-ner-2017-06-09/stanford-ner-2017-06-09/classifiers/english.muc.7class.distsim.crf.ser.gz',
               'C:/stanford-ner-2017-06-09/stanford-ner-2017-06-09/stanford-ner.jar') 
def normalize_text(text, lemmatize = True,remove_stop = None):

    import re
    text = re.compile(r'\W+', re.UNICODE).split(text) #remove non alphanumeric
    from nltk.corpus import stopwords
    stop = stopwords.words('english')
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    
    from itertools import groupby

    tagged_text = st.tag(text) 
    
    text_with_tags_added = []
    
    for tag, chunk in groupby(tagged_text, lambda x:x[1]):
        if tag != "O":
            #print("%-12s"%tag, " ".join(w for w, t in chunk))  
            text_with_tags_added.append((u'#'+tag))
        else:
            for w, t in chunk:
                text_with_tags_added.append(w.lower())    
    
    clean_text = [] 
    for word in text_with_tags_added:
        if word.isdigit():
            clean_text.append(u'#NUMBER')
        elif not(remove_stop == True and word in stop):
            clean_text.append(word)
            
    if lemmatize == True:        
        " ".join([ lemmatizer.lemmatize(word) for word in clean_text if word[0]!='#'])
    else:
        " ".join([ word for word in clean_text])
    
    return clean_text                    
##################################################################################
#def normalize_text(text, lemmatize = None,remove_stop = None):
#
#    import re
#    text = re.compile(r'\W+', re.UNICODE).split(text) #remove non alphanumeric
#    from nltk.corpus import stopwords
#    stop = stopwords.words('english')
#    from nltk.stem import WordNetLemmatizer
#    lemmatizer = WordNetLemmatizer()
#    
#    
#     
#    
#    clean_text = [] 
#    for word in text:
#        if word.isdigit():
#            clean_text.append(u'#NUMBER')
#        elif not(remove_stop == True and word in stop):
#            clean_text.append(word)
#            
#    if lemmatize == True:        
#        " ".join([ lemmatizer.lemmatize(word) for word in clean_text])
#    else:
#        " ".join([ word for word in clean_text])
#    
#    return clean_text                             
##################################################################################        
############################### Start Execution Point #############################        
###################################################################################        

from time import gmtime, strftime
print strftime("%Y-%m-%d %H:%M:%S", gmtime())
# Uni
machinename = 'maj27'



j=0
for i in range(10):
    dataset = str(i+1)
    #mydir = 'C:/Users/'+machinename+'/New folder/Dropbox/PhD Brighton/Dataset/healthnewsreview_org/Classified News/Training'
    train_dir = 'C:/Users/'+machinename+'/New folder/Dropbox/PhD Brighton/Dataset/healthnewsreview_org/Well done 5 and 10 inverted/Classified Story/Criteria '+dataset+'/Train+val'
    test_dir = 'C:/Users/'+machinename+'/New folder/Dropbox/PhD Brighton/Dataset/healthnewsreview_org/Well done 5 and 10 inverted/Classified Story/Criteria '+dataset+'/Testing'
    #test_dir = 'C:/Users/'+machinename+'/New folder/Dropbox/PhD Brighton/Dataset/healthnewsreview_org/NA is negative old/Classified News/Criteria '+dataset+''
    
 
    preprocessed = 'C:/Users/'+machinename+'/New folder/Dropbox/PhD Brighton/Dataset/healthnewsreview_org/Well done 5 and 10 inverted/Classified Story/Criteria '+dataset+'/data_2.p'
    
    train_Corpus = CategorizedPlaintextCorpusReader(train_dir, r'(?!\.).*\.txt', cat_pattern=r'(\w+)/*')
    
    train_documents = [(list(train_Corpus.words(fileid)),category)
                  for category in train_Corpus.categories()
                  for fileid in train_Corpus.fileids(category)]
    
         
    only_docs = [' '.join(doc[:1000]) for (doc,category) in train_documents]
    only_docs = [' '.join(normalize_text(document,lemmatize = True,remove_stop = None)) for document in only_docs]
    
     #######################################################################################             
    train_labels = [category for (doc,category) in train_documents]
    train_binary_labels = [1 if i=='pos' else 0 for i in train_labels]
    
    #train_data, test_data, train_labels, test_labels = train_test_split(only_docs, binary_labels,test_size=.15)
    train_data = only_docs
    train_labels = train_binary_labels
    
  ############################ setup test set ##########################################
    
    test_Corpus = CategorizedPlaintextCorpusReader(test_dir, r'(?!\.).*\.txt', cat_pattern=r'(\w+)/*')
    
    test_documents = [(list(test_Corpus.words(fileid)),category)
                  for category in test_Corpus.categories()
                  for fileid in test_Corpus.fileids(category)]
    
    test_fileIds = [ fileid for fileid in test_Corpus.fileids()]    
    
    #test_only_docs = [' '.join(doc) for (doc,category) in test_documents]
    # or use only the first 1000 words (to get rid of comments on some articles)
    test_only_docs = [' '.join(doc[:1000]) for (doc,category) in test_documents]
    test_only_docs = [' '.join(normalize_text(document,lemmatize = True,remove_stop = None)) for document in test_only_docs]
    
    
    test_labels = [category for (doc,category) in test_documents]
    test_binary_labels = [1 if i=='pos' else 0 for i in test_labels]
    
    
    #train_data, test_data, train_labels, test_labels = train_test_split(only_docs, binary_labels,test_size=.15)
    test_data = test_only_docs
    test_labels = test_binary_labels


   # Saving the objects:
    with open(preprocessed, 'w') as f:  
        pickle.dump([train_data, train_labels, test_data,test_labels,test_fileIds], f)


print strftime("%Y-%m-%d %H:%M:%S", gmtime())