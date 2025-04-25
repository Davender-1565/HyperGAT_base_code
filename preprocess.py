print("Preprocessing started")
from nltk.corpus import stopwords
import nltk
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from utils import clean_str, show_statisctic, clean_document, clean_str_simple_version
import collections
from collections import Counter
import random
import numpy as np
import pickle
import json
from nltk import tokenize
from sklearn.utils import class_weight
import torch

def read_file(dataset, LDA=True):
    
    doc_content_list = []
    doc_sentence_list = []
    f = open('data/' + dataset + '_corpus.txt', 'rb')

    for line in f.readlines():
        doc_content_list.append(line.strip().decode('latin1'))
        doc_sentence_list.append(tokenize.sent_tokenize(clean_str_simple_version(doc_content_list[-1], dataset)))
    f.close()

    doc_content_list = clean_document(doc_sentence_list, dataset)

    max_num_sentence = show_statisctic(doc_content_list)

    doc_train_list_original = []
    doc_test_list_original = []
    labels_dic = {}
    label_count = Counter()

    i = 0
    f = open('data/' + dataset + '_labels.txt', 'r')
    lines = f.readlines()
    for line in lines:
        temp = line.strip().split("\t")
        if temp[1].find('test') != -1:
            doc_test_list_original.append((doc_content_list[i],temp[2]))
        elif temp[1].find('train') != -1:
            doc_train_list_original.append((doc_content_list[i],temp[2]))
        #print(temp)
        if not temp[2] in labels_dic:
            labels_dic[temp[2]] = len(labels_dic)
        label_count[temp[2]] += 1
        i += 1

    f.close()
    print(label_count)

    word_freq = Counter()
    word_set = set()
    for doc_words in doc_content_list:
        for words in doc_words:
            for word in words:
                word_set.add(word)
                word_freq[word] += 1

    vocab = list(word_set)
    vocab_size = len(vocab)

    vocab_dic = {}
    for i in word_set:
        vocab_dic[i] = len(vocab_dic) + 1

    print('Total_number_of_words: ' + str(len(vocab)))
    print('Total_number_of_categories: ' + str(len(labels_dic)))

    doc_train_list = []
    doc_test_list = []

    for doc,label in doc_train_list_original:
        temp_doc = []
        for sentence in doc:
            temp = []
            for word in sentence:
                temp.append(vocab_dic[word])
            temp_doc.append(temp)
        doc_train_list.append((temp_doc,labels_dic[label]))

    for doc,label in doc_test_list_original:
        temp_doc = []
        for sentence in doc:
            temp = []
            for word in sentence:
                temp.append(vocab_dic[word])
            temp_doc.append(temp)
        doc_test_list.append((temp_doc,labels_dic[label]))

    keywords_dic = {}
    if LDA:
        keywords_dic_original = pickle.load(open('data/' + dataset + '_LDA.p', "rb" ))
    
        for i in keywords_dic_original:
            if i in vocab_dic:
                keywords_dic[vocab_dic[i]] = keywords_dic_original[i]

    train_set_y = [j for i,j in doc_train_list]
    
    unique_classes = np.unique(train_set_y)
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,  # Use all possible classes
        y=train_set_y
    )

    full_class_weights = np.ones(len(labels_dic))
    full_class_weights[unique_classes] = class_weights
    print(f"Class weights computed for {len(class_weights)} classes")

    return doc_content_list, doc_train_list, doc_test_list, vocab_dic, labels_dic, max_num_sentence, keywords_dic, class_weights

def loadGloveModel(gloveFile, vocab_dic, matrix_len):
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    gloveModel = {}
    glove_embedding_dimension = 0
    
    # First pass to get dimension
    first_line = f.readline().split()
    glove_embedding_dimension = len(first_line[1:])
    f.seek(0)  # Rewind
    
    # Initialize with +1 to account for padding index
    weights_matrix = np.zeros((matrix_len + 1, glove_embedding_dimension))  # Changed
    weights_matrix[0] = np.zeros((glove_embedding_dimension, ))
    
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        gloveModel[word] = embedding
    
    words_found = 0
    for word in vocab_dic:
        # Add +1 to index to account for padding at 0
        if word in gloveModel:
            weights_matrix[vocab_dic[word] + 1] = gloveModel[word]  # Changed
            words_found += 1
        else:
            weights_matrix[vocab_dic[word] + 1] = gloveModel.get('the', np.random.normal(scale=0.6, size=glove_embedding_dimension))  # Changed

    f.close()
    print("Total ", len(vocab_dic), " words")
    print("Done.",words_found," words loaded from", gloveFile)
    
    # Convert to torch tensor
    weights_matrix = torch.FloatTensor(weights_matrix)
    assert weights_matrix.dim() == 2, "Embedding matrix must be 2-dimensional"
    return weights_matrix
print("Preprocessing Ended")