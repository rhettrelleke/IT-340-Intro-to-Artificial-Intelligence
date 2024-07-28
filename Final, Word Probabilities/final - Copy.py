# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 20:13:57 2023

@author: Rhett Relleke

IT 340 Final

POS Tagging using Viterbi Algorithm
"""

import matplotlib.pyplot as plt
import numpy as np


def parse_tagged_sentences(file_path):
    sentences = []
    word_counts = {}  #initialize word counts here

    with open(file_path, 'r') as file:
        for line in file:
            words_tags = line.strip().split()
            sentence = []
            for word_tag in words_tags:
                if '/' in word_tag and not word_tag.endswith('$$'):
                    word, tag = word_tag.rsplit('/', 1)
                else:
                    word, tag = word_tag, None

                sentence.append((word, tag))
                #update word counts
                word_counts[word] = word_counts.get(word, 0) + 1
            sentences.append(sentence)

    return sentences, word_counts


def calculate_emission_prob(word_tag_count, tag_count, word, tag, tag_set):
    #calculate the emission probability with Laplace smoothing
    return (word_tag_count.get((word, tag), 0) + 1) / (tag_count.get(tag, 0) + len(tag_set))

def calculate_transition_prob(tag_bigram_count, tag_count, prev_tag, tag, tag_set):
    #calculate the transition probability with Laplace smoothing
    return (tag_bigram_count.get((prev_tag, tag), 0) + 1) / (tag_count.get(prev_tag, 0) + len(tag_set))

def log_probability(prob):
    #check if prob is an array and handle accordingly
    if isinstance(prob, np.ndarray):
        log_probs = np.log(prob)
        log_probs[prob <= 0] = float('-inf')  #replace log(0) or log(negative) with -inf
        return log_probs
    else:
        #for single number
        return np.log(prob) if prob > 0 else float('-inf')


def calculate_probabilities(tagged_sentences):
    word_tag_count, tag_bigram_count, tag_counts = {}, {}, {'##': len(tagged_sentences)}
    for sentence in tagged_sentences:
        prev_tag = '##'
        for word, tag in sentence:
            if tag is not None:
                word_tag_count[(word, tag)] = word_tag_count.get((word, tag), 0) + 1
                tag_bigram_count[(prev_tag, tag)] = tag_bigram_count.get((prev_tag, tag), 0) + 1
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
                prev_tag = tag
            tag_bigram_count[(tag, '$$')] = tag_bigram_count.get((tag, '$$'), 0) + 1
    return word_tag_count, tag_bigram_count, tag_counts

def viterbi_algorithm(sentence, tag_set, word_tag_count, tag_count, tag_bigram_count):
    #exclude '$$' from tag_set for initialization and recursion steps
    working_tag_set = tag_set - {'$$'}

    #initialization
    viterbi = [{tag: log_probability(calculate_transition_prob(tag_bigram_count, tag_count, '##', tag, tag_set)) + \
                log_probability(calculate_emission_prob(word_tag_count, tag_count, sentence[0], tag, tag_set)) \
                for tag in working_tag_set}]
    backpointer = [{}]

    #recursion step over the sentence
    for word_index in range(1, len(sentence)):
        viterbi.append({})
        backpointer.append({})

        for tag in working_tag_set:
            max_tr_prob, best_prev_tag = max(
                ((viterbi[word_index - 1][prev_tag] + log_probability(calculate_transition_prob(tag_bigram_count, tag_count, prev_tag, tag, tag_set)), prev_tag)
                 for prev_tag in working_tag_set),
                key=lambda x: x[0]
            )
            viterbi[word_index][tag] = max_tr_prob + log_probability(calculate_emission_prob(word_tag_count, tag_count, sentence[word_index], tag, tag_set))
            backpointer[word_index][tag] = best_prev_tag

    #termination step for the end of the sentence
    last_word_probabilities = {tag: viterbi[-1][tag] + log_probability(calculate_transition_prob(tag_bigram_count, tag_count, tag, '$$', tag_set)) for tag in working_tag_set}
    last_tag = max(last_word_probabilities, key=last_word_probabilities.get)
    best_tag_sequence = [last_tag]

    #backtracking to find the best tag sequence
    for word_index in range(len(sentence) - 1, 0, -1):
        last_tag = backpointer[word_index][last_tag]
        best_tag_sequence.insert(0, last_tag)

    return best_tag_sequence

#function to extract words from a sentence string
def extract_words(sentence_str):
    return sentence_str.split()

#function to compare predicted tags with actual tags
def compare_tags(predicted, actual):
    correct = sum(p == a for p, a in zip(predicted, actual))
    total = len(actual)
    return correct, total

tagged_sentences, word_counts = parse_tagged_sentences('tagged_sentences.txt')
word_tag_count, tag_bigram_count, tag_counts = calculate_probabilities(tagged_sentences)
tag_set = set(tag_counts.keys()) - {None, '##', '$$'}

#creating the Word Frequency graph
selected_words = ['family', 'guy', 'peter', 'griffin']
selected_word_counts = [word_counts.get(word, 0) for word in selected_words]

plt.figure(figsize=(10, 5))
plt.bar(selected_words, selected_word_counts, color='blue')
plt.xlabel('Words')
plt.ylabel('Count')
plt.title('Word Frequency')
for i, count in enumerate(selected_word_counts):
    plt.text(i, count + 3, str(count), ha='center')
plt.show()

#remove the '##' entries from tag_counts
if '##' in tag_counts:
    del tag_counts['##']

#sort the tags by count in descending order
sorted_tags = sorted(tag_counts.items(), key=lambda item: item[1], reverse=True)

#separate the tags and their counts for plotting
tags, counts = zip(*sorted_tags)

#creating the POS Tag distribution graph
plt.figure(figsize=(10, 8))
plt.barh(tags, counts, color='blue')
plt.xlabel('Counts')
plt.title('POS Tag Distribution')
for index, value in enumerate(counts):
    plt.text(value, index, str(value))
plt.tight_layout()
plt.show()

#initialize the count for the start symbol '##'
tag_counts['##'] = len(tagged_sentences)

for sentence in tagged_sentences:
    prev_tag = '##'  #start symbol
    for word, tag in sentence:
        if tag is not None:
            #increment the word-tag pair count
            word_tag_count[(word, tag)] = word_tag_count.get((word, tag), 0) + 1
            
            #increment the tag bigram count
            tag_bigram_count[(prev_tag, tag)] = tag_bigram_count.get((prev_tag, tag), 0) + 1
            
            #set the current tag as the previous tag for the next iteration
            prev_tag = tag

#add counts for the end symbol '$$' following the last tag of each sentence
for tag in tag_counts:
    tag_bigram_count[(tag, '$$')] = tag_bigram_count.get((tag, '$$'), 0) + tag_counts.get(tag, 0)

#sentences for testing
test_sentences = [
    ("nice !", ['ADJ', '.']),
    ("good lord !", ['ADJ', 'NOUN', '.']),
    ("how are you ?", ['ADV', 'VERB', 'PRON', '.']),
    ("they can fish .", ['PRON', 'VERB', 'NOUN', '.']),
    ("she is on diet .", ['PRON', 'VERB', 'ADP', 'NOUN', '.']),
    ("we got kicked out .", ['PRON', 'VERB', 'VERB', 'PRT', '.']),
    ("there is no free food .", ['PRT', 'VERB', 'DET', 'ADJ', 'NOUN', '.']),
    ("no need to worry about it at all !", ['DET', 'VERB', 'PRT', 'VERB', 'ADP', 'PRON', 'ADP', 'PRT', '.']),
    ("how come the professor let him pass ?", ['ADV', 'VERB', 'DET', 'NOUN', 'VERB', 'PRON', 'VERB', '.']),
    ("maybe we shouldn't tell our parents that we didn't go .", ['ADV', 'PRON', 'VERB', 'VERB', 'DET', 'NOUN', 'ADP', 'PRON', 'VERB', 'VERB', '.']),
    ("how many people are there in the conference room ?", ['ADV', 'ADJ', 'NOUN', 'VERB', 'PRT', 'ADP', 'DET', 'NOUN', 'NOUN', '.']),
    ("not sure if we can get there on time or not , so we should leave early .", ['ADV', 'ADJ', 'ADP', 'PRON', 'VERB', 'VERB', 'PRT', 'ADP', 'NOUN', 'CONJ', 'ADV', '.', 'ADV', 'PRON', 'VERB', 'VERB', 'ADV', '.']),
    ("speaking of that , we'll let you handle it .", ['VERB', 'ADP', 'PRON', '.', 'PRT', 'VERB', 'PRON', 'VERB', 'PRON', '.']),
    ("remember : we are not here to kill time !", ['VERB', '.', 'PRON', 'VERB', 'ADV', 'ADV', 'PRT', 'VERB', 'NOUN', '.']),
    ("no offense but the result is not what we want . . .", ['DET', 'NOUN', 'CONJ', 'DET', 'NOUN', 'VERB', 'ADV', 'DET', 'PRON', 'VERB', '.', '.', '.']),
    ("2 examples are far from enough , you should try , at least , 3 more .", ['NUM', 'NOUN', 'VERB', 'ADV', 'ADP', 'ADJ', '.', 'PRON', 'VERB', 'VERB', '.', 'ADP', 'ADJ', '.', 'NUM', 'ADJ', '.']),
    ("who would have thought we had to do remote learning for the past 2 years ? !", ['PRON', 'VERB', 'VERB', 'VERB', 'PRON', 'VERB', 'PRT', 'VERB', 'PRT', 'VERB', 'ADP', 'DET', 'ADJ', 'NUM', 'NOUN', '.', '.']),
    ("had the ball found the net , the goal would have been ruled out .", ['VERB', 'DET', 'NOUN', 'VERB', 'DET', 'NOUN', '.', 'DET', 'NOUN', 'VERB', 'VERB', 'VERB', 'VERB', 'PRT', '.']),
    ("and the result of the analysis shows that people are currently not interested in buying their products .", ['CONJ', 'DET', 'NOUN', 'ADP', 'DET', 'NOUN', 'VERB', 'ADP', 'NOUN', 'VERB', 'ADV', 'ADV', 'VERB', 'ADP', 'VERB', 'DET', 'NOUN', '.']),
    ("since the final project is hard , we should ask the professor for help !", ['ADP', 'DET', 'ADJ', 'NOUN', 'VERB', 'ADV', '.', 'PRON', 'VERB', 'VERB', 'DET', 'NOUN', 'ADP', 'NOUN', '.']),
    ]


#initialize counters for accuracy calculation
total_correct = 0
total_tags = 0

#test each sentence
for sentence_str, correct_tags in test_sentences:
    words = extract_words(sentence_str)
    predicted_tags = viterbi_algorithm(words, tag_set, word_tag_count, tag_counts, tag_bigram_count)

    #compare and count correct predictions
    correct, total = compare_tags(predicted_tags, correct_tags)
    total_correct += correct
    total_tags += total

    #print results
    print(f"Sentence: {sentence_str}")
    print(f"Predicted Tags: {predicted_tags}")
    print(f"Actual Tags: {correct_tags}")
    print(f"Correctly Predicted: {correct}/{total}")
    print("---------------------------------------------------------------")

#calculate and print overall accuracy
accuracy = total_correct / total_tags if total_tags > 0 else 0
print(f"Overall Accuracy: {accuracy:.2f}")


