import os
import pandas as pd
import numpy as np
from numpy import interp
import string
import math
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
porter_stemmer = PorterStemmer()


def get_paths(folder):
    """
    Makes a list of all the paths that are in a folder
    
    :param folder: the path of a folder
    :return  Returns a list of all the pathnames
    """
    current_directory = os.getcwd()
    storiesPath = []
    for file in os.listdir(folder):
        storiesPath.append(os.path.join(current_directory + '/' + folder, file))
    return storiesPath


def load_data(pathset):
    """
    Loads the data into a dataframe
    :param pathset:  A list of paths
    :returns  A dataframe that only keeps stories with more than 10 sentences. Also, splits up stories and highlights
    """

    keepPath = []
    stories = []

    df = pd.DataFrame(columns = ['Path', 'RawStory'])
    #i = 0
    for path in pathset:
        text = open(path, "r").read()
        tok_sent = sent_tokenize(text)
        if len(tok_sent) >= 10: # If the articles is at least 10 sentences long and
            if '(CNN)' in tok_sent[0]: # Only consider articles formatted "(CNN)"
                keepPath.append(path)
                stories.append(text)
    df['Path'] = keepPath
    df['RawStory'] = stories
    
    # Split up text and highlights
    df['Story'] = [x.split("@highlight",1)[0] for x in df.RawStory]
    df['Highlights'] = [x.split("@highlight",1)[1].replace('\n', '').split('@highlight') for x in df.RawStory]

    return df


def preprocess(df):
    """
    Preprocess text. For each story it does the following:
    - 
    
    :param df: a dataframe which contains paths and stories without highlights
    :return .csv files with tokenized and preprocesses sentences and words
    :return the path of the preprocessed files 
    """

    # Create a directory where we store preprocessed files, but first check if it already exists
    newpath = r'test_texts/processed_stories'
    if not os.path.exists(newpath): 
        os.makedirs(newpath)
    

    for story,path in zip(df['Story'],df['Path']):

        df = pd.DataFrame(columns = ['Label', 'Sentence', 'Words'])

        sent_position = []
        words = []
        sents = []
        i = 1

        sentences = sent_tokenize(story)

        # We've taken all articles with (CNN) in it. Now remove it
        remove = '.*?\(CNN\)(\s+)?(--)?(\s+)?'
        sentences[0] = re.sub(remove, '', sentences[0])

        for sentence in sentences:

            # Tokenize words
            current_words = word_tokenize(sentence)

            # Then remove punctuation
            current_words = [x.casefold() for x in current_words if not re.fullmatch('[' + string.punctuation + ']+', x)]
            # Remove stopwords and stem
            current_words = [porter_stemmer.stem(word) for word in current_words if not word in stopwords.words('english')]

            # Now make a list of these words
            words.append(current_words)

            # But keep original setnences
            sents.append(sentence)

            # Finally label the sentence according to the order in which it appears
            sent_position.append(i)
            i += 1

        df['Label'] = sent_position
        df['Sentence'] = sents
        df['Words'] = words
        
        fileName = os.path.basename(path) # also contains extension
        fileName = fileName.split('.',1)[0] # deletes extension

        filePath = newpath + '/' + fileName
        df.to_csv(filePath + '.csv')
        
    return newpath
