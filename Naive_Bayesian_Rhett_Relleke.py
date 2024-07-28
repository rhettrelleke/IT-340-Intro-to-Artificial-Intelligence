# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 16:49:38 2023

@author: Rhett
"""

from os import walk
import numpy as np

seed = 12345
number_of_training_emails = 500

ham_dir = 'Ham'
All_Ham = []
for (dirpath, dirnames, filenames) in walk(ham_dir):
    if filenames != []:
        for ham in filenames:
            with open(dirpath+'\\'+ham) as f:
                #read the file as a big string
                All_Ham.append(f.read())
                
spam_dir = 'Spam'
All_Spam = []
for (dirpath, dirnames, filenames) in walk(spam_dir):
    if filenames != []:
        for spam in filenames:
            with open(dirpath+'\\'+spam,encoding='latin-1') as f:
                #read the file as a big string
                All_Spam.append(f.read())           
                

randomState = np.random.RandomState(seed)
randomState.shuffle(All_Ham)
randomState.shuffle(All_Spam)

training_data = All_Ham[:number_of_training_emails] + All_Spam[:number_of_training_emails]
testing_data = All_Ham[number_of_training_emails:] + All_Spam[number_of_training_emails:] 

#important: use .split() to separate the words 
def data_loader(training_data_ham, training_data_spam, testing_data_ham, testing_data_spam):
    #Concatenate ham + spam training data and labels
    training_emails = training_data_ham + training_data_spam
    training_labels = np.array([0] * len(training_data_ham) + [1] * len(training_data_spam))

    #Concatenate ham + spam testing data and labels
    testing_emails = testing_data_ham + testing_data_spam
    testing_labels = np.array([0] * len(testing_data_ham) + [1] * len(testing_data_spam))
    
    
    training_emails = [email.split() for email in training_emails]
    testing_emails = [email.split() for email in testing_emails]

    return training_emails, training_labels, testing_emails, testing_labels

#Train the Naive Bayes classifier
def train_naive_bayes(training_emails, training_labels):
    #Make a dictionary to hold the probability of each word given spam and ham
    word_probs = {}
    num_spam = sum(training_labels)
    num_ham = len(training_labels) - num_spam
    
    #Initialize word counts
    for email in training_emails:
        for word in email:
            if word not in word_probs:
                word_probs[word] = {'Spam': 1, 'Ham': 1}  #Laplace smoothing

    #Count word occurrences in spam and ham emails
    for i, email in enumerate(training_emails):
        for word in email:
            if training_labels[i] == 1:
                word_probs[word]['Spam'] += 1
            else:
                word_probs[word]['Ham'] += 1

    #Calculate probabilities
    for word, counts in word_probs.items():
        counts['Spam'] = counts['Spam'] / (num_spam + 2)  #Laplace smoothing
        counts['Ham'] = counts['Ham'] / (num_ham + 2)  #Laplace smoothing
    
    return word_probs, num_spam / len(training_emails), num_ham / len(training_emails)

#Classify using the trained classifier
def classify(email, word_probs, p_spam, p_ham):
    spam_prob = np.log(p_spam)
    ham_prob = np.log(p_ham)
    
    for word in email:
        if word in word_probs:
            spam_prob += np.log(word_probs[word]['Spam'])
            ham_prob += np.log(word_probs[word]['Ham'])
        else:
            spam_prob += np.log(1 / (p_spam + 2))
            ham_prob += np.log(1 / (p_ham + 2))

    return 1 if spam_prob > ham_prob else 0

#Evaluate classifier
def evaluate(test_emails, test_labels, word_probs, p_spam, p_ham):
    predictions = [classify(email, word_probs, p_spam, p_ham) for email in test_emails]
    
    accuracy = np.mean([pred == label for pred, label in zip(predictions, test_labels)])
    precision = np.mean([pred == label for pred, label in zip(predictions, test_labels) if pred == 1])
    recall = np.mean([pred == label for pred, label in zip(predictions, test_labels) if label == 1])
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

    return accuracy, precision, recall, f1_score

def main():
    training_data_ham = All_Ham[:number_of_training_emails]
    training_data_spam = All_Spam[:number_of_training_emails]
    testing_data_ham = All_Ham[number_of_training_emails:]
    testing_data_spam = All_Spam[number_of_training_emails:]
    training_emails, training_labels, testing_emails, testing_labels = data_loader(training_data_ham, training_data_spam, testing_data_ham, testing_data_spam)
    word_probs, p_spam, p_ham = train_naive_bayes(training_emails, training_labels)
    accuracy, precision, recall, f1_score = evaluate(testing_emails, testing_labels, word_probs, p_spam, p_ham)
    
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1_score}")

if __name__ == "__main__":
    main()