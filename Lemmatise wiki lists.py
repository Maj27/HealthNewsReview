



# -*- coding: utf-8 -*-
"""
Created on  May 1 18:58:00 2018

@author: maj27

This code is to lemmatise the wiki lists

"""





###############################################################################
                 ###############################################################################
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer() 

def read_words(words_file):
    with open(words_file, 'r') as f:
        words_list = []
        for word in f:
            words_list.append(lemmatizer.lemmatize(word.rstrip()))
            
        return words_list

               
      
        
###############################################################################
def lemmatise_lists():
    
    
    comparitive_forms = read_words('C:/Users/maj27/Downloads/wiktionarylists/wiktionarylists/comparative_forms.txt')
    modal_adverbs = read_words('C:/Users/maj27/Downloads/wiktionarylists/wiktionarylists/modal_adverbs.txt')
    act_adverbs = read_words('C:/Users/maj27/Downloads/wiktionarylists/wiktionarylists/act_adverbs.txt')
    manner_adverbs = read_words('C:/Users/maj27/Downloads/wiktionarylists/wiktionarylists/manner_adverbs.txt')
    superlative_forms = read_words('C:/Users/maj27/Downloads/wiktionarylists/wiktionarylists/superlative_forms.txt')
    numerals = read_words('C:/Users/maj27/Downloads/wiktionarylists/wiktionarylists/numerals.txt')
    degree_adverbs = read_words('C:/Users/maj27/Downloads/wiktionarylists/wiktionarylists/degree_adverbs.txt')
    auxiliary_verbs = read_words('C:/Users/maj27/Downloads/wiktionarylists/wiktionarylists/auxiliary_verbs.txt')
    

    for word in comparitive_forms:
        with open('C:/Users/maj27/Downloads/wiktionarylists/wiktionarylists/lemmatised/comparative_forms.txt', 'a') as the_file:
             the_file.write(word + '\n')
     
    for word in modal_adverbs:
        with open('C:/Users/maj27/Downloads/wiktionarylists/wiktionarylists/lemmatised/modal_adverbs.txt', 'a') as the_file:
             the_file.write(word + '\n')

    for word in act_adverbs:
        with open('C:/Users/maj27/Downloads/wiktionarylists/wiktionarylists/lemmatised/act_adverbs.txt', 'a') as the_file:
             the_file.write(word + '\n')

    for word in manner_adverbs:
        with open('C:/Users/maj27/Downloads/wiktionarylists/wiktionarylists/lemmatised/manner_adverbs.txt', 'a') as the_file:
             the_file.write(word + '\n')

    for word in superlative_forms:
        with open('C:/Users/maj27/Downloads/wiktionarylists/wiktionarylists/lemmatised/superlative_forms.txt', 'a') as the_file:
             the_file.write(word + '\n')

    for word in numerals:
        with open('C:/Users/maj27/Downloads/wiktionarylists/wiktionarylists/lemmatised/numerals.txt', 'a') as the_file:
             the_file.write(word + '\n')
             
    for word in degree_adverbs:
        with open('C:/Users/maj27/Downloads/wiktionarylists/wiktionarylists/lemmatised/degree_adverbs.txt', 'a') as the_file:
             the_file.write(word + '\n')
             
    for word in auxiliary_verbs:
        with open('C:/Users/maj27/Downloads/wiktionarylists/wiktionarylists/lemmatised/auxiliary_verbs.txt', 'a') as the_file:
             the_file.write(word + '\n')


from time import gmtime, strftime
print strftime("%Y-%m-%d %H:%M:%S", gmtime())
# Uni
machinename = 'maj27'

lemmatise_lists()

             
print strftime("%Y-%m-%d %H:%M:%S", gmtime())