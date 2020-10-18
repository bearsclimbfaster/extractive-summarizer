from preprocessing import *

def get_PS(processedData):
    """
    Calculate the scores for the feature POSITION of SENTENCE using the equation:
    PS = 1 - (i - 1)/N
    
    :param processedData: a dataframe which contains preprocesses sentences and words 
    :return vector containing the scores of those sentences according to the feature POSITION of SENTENCE
    """
    N = len(processedData.Label)

    PS_scores = []
    for label in processedData.Label:
        PS_scores.append(1 - (label - 1)/N)
    processedData['PS Scores'] = PS_scores

    return processedData

# -------------------------------------------------------------------------------------------------------------

def get_LS(processedData):
    """
    Calculate the scores for the feature LENGTH of SENTENCE using the equation:
    LS = #words in S / #words in longest sentence
    
    :param processedData: a dataframe which contains preprocesses sentences and words 
    :return same dataframe but now containing the scores of those sentences according to the feature LENGTH of SENTENCE
    """
    maxWords = max([len(processedData.Words[i]) for i in range(0,len(processedData.Words))])
    LS_scores = []
    for words in processedData.Words:
        LS_scores.append(len(words)/maxWords)
    processedData['LS Scores'] = LS_scores

    return processedData


# https://towardsdatascience.com/text-summarization-using-tf-idf-e64a0644ace3

def get_word_freq(processedData):
    """
    Calculates the frequency of unique words within a specific sentence (build vocabulary)
    
    :param processedData: a dataframe which contains preprocesses sentences and words 
    :return dictionary where the key is the sentence label and the value is another vocabulary with
     the frequency of each word contained in the sentence
    """
    vocabulary = {}
    for label, tokenizedSent in zip(processedData.Label, processedData.Words):
        freq = {}
        for word in tokenizedSent:
            if word in freq:
                freq[word] += 1
            else:
                freq[word] = 1
        vocabulary[label] = freq

    return vocabulary

# -------------------------------------------------------------------------------------------------------------

def get_TF(vocabulary):
    """
    Calculates the Term Frequency (TF) given the frequency dictionary
    
    :param vocabulary: see above
    :return another dictionary where the values this time are TF's
    """
    tf_matrix = {}

    for sentLabel, freq_table in vocabulary.items():
        tf_table = {}

        count_words_in_sentence = len(freq_table)
        for word, count in freq_table.items():
            tf_table[word] = count / count_words_in_sentence

        tf_matrix[sentLabel] = tf_table

    return tf_matrix

# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------

def create_documents_per_words(vocabulary):
    """
    Calculates in how many sentences a word appears
    
    :param vocabulary: see above
    :return another dictionary where the key are words and the values the no of sentences in which they appear
    """
    no_word_per_doc_table = {}

    for sent, freq_table in vocabulary.items():
        for word, count in freq_table.items():
            if word in no_word_per_doc_table:
                no_word_per_doc_table[word] += 1
            else:
                no_word_per_doc_table[word] = 1

    return no_word_per_doc_table

# -------------------------------------------------------------------------------------------------------------

def get_IDF(freq_matrix, word_per_doc_table, total_documents):
    """
    Calculates Inverse Document Frequency (IDF)
    
    :param vocabulary: see above
    :return another dictionary where the key are words and the values the no of sentences in which they appear
    """

    idf_matrix = {}

    for sent, freq_table in freq_matrix.items():
        idf_table = {}

        for word in freq_table.keys():
            idf_table[word] = math.log10(total_documents / float(word_per_doc_table[word]))

        idf_matrix[sent] = idf_table

    return idf_matrix

# -------------------------------------------------------------------------------------------------------------

def create_tf_idf_matrix(tf_matrix, idf_matrix):
    tf_idf_matrix = {}

    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):

        tf_idf_table = {}

        for (word1, value1), (word2, value2) in zip(f_table1.items(),
                                                    f_table2.items()):  # here, keys are the same in both the table
            tf_idf_table[word1] = float(value1 * value2)

        tf_idf_matrix[sent1] = tf_idf_table

    return tf_idf_matrix

# -------------------------------------------------------------------------------------------------------------

def score_TF_IDF(tf_idf_matrix) -> dict:
    """
    score a sentence by its word's TF
    Basic algorithm: adding the TF frequency of every non-stop word in a sentence divided by total no of words in a sentence.
    :rtype: dict
    """

    sentenceValue = {}

    for sent, f_table in tf_idf_matrix.items():
        total_score_per_sentence = 0

        count_words_in_sentence = len(f_table)
        for word, score in f_table.items():
            total_score_per_sentence += score
        if count_words_in_sentence == 0:
            sentenceValue[sent] = 0
        else:
            sentenceValue[sent] = total_score_per_sentence / count_words_in_sentence

    return sentenceValue

# -------------------------------------------------------------------------------------------------------------

def map_scores(scores):
    """
    Takes TF IDF scores and redistribute them between 0 and 1, where 1 represents the highest score and 0 the lowest
    :param scores: a dictionary with TF IDF scores, where key is sentence label and value is score 
    return: re-mapped scores
    """

    minScore = min(scores.values())
    maxScore = max(scores.values())

    mappedScores = {}
    for sent, score in scores.items():
        mappedScores[sent] = interp(score,[minScore,maxScore],[0,1])
    
    return mappedScores

# -------------------------------------------------------------------------------------------------------------

def get_TF_IDF(processedData):

    tot_sentences = len(processedData['Words'])

    # Build a vocabulary per sentence
    vocab = get_word_freq(processedData)

    # Find their Term Frequency (TF)
    TF = get_TF(vocab)

    # See in how many sentence each word appears
    no_word_per_doc_table = create_documents_per_words(vocab)

    # Calculate Inverse Document Frequency (IDF)
    IDF = get_IDF(vocab, no_word_per_doc_table,tot_sentences)

    # Use TF and IDF to calculate TF-IDF values
    TF_IDF_matrix = create_tf_idf_matrix(TF,IDF)

    # Use TF-IDF values to give scores to the sentences
    scores = score_TF_IDF(TF_IDF_matrix)

    # Remap scores from 0 to 1
    mappedScores = map_scores(scores)

    # Make another column in 'processedData' with the score of each sentence
    processedData['TF-IDF Scores'] = mappedScores.values()

    return processedData


def find_propernouns(processedData):
    NNP_scores = []
    for sent in processedData['Sentence']:
        propernouns = [word for word,pos in pos_tag(sent.split()) if pos == 'NNP' or pos == 'NNPS']
        NNP_scores.append(1 if propernouns != [] else 0)
    
    processedData['ProperNouns'] = NNP_scores
    return processedData

def get_summary(processedData, percentage, w_PS, w_LS, w_TF_IDF, w_NNP):
    """
    Given some processed data for which the scores of the different features have been computed,
    perform a weighted sum of these and output the k best sentences
    
    :param processedData: a dataframe which contains preprocesses sentences and words 
    :return data including weighted sum and summary of k sentences with highest score
    """

    noSentences = len(processedData['Sentence'])
    k = int(round(noSentences * percentage/100))

    processedData['Weighted Sum'] = w_LS * processedData['LS Scores'] +\
    w_PS * processedData['PS Scores'] + w_TF_IDF * processedData['TF-IDF Scores'] +\
    w_NNP * processedData['ProperNouns']
    
    sortedDF = processedData.sort_values('Weighted Sum')
    sortedDF = sortedDF.reset_index(drop=True)

    sortedDF = processedData.sort_values('Weighted Sum', ascending=False)
    sortedDF = sortedDF.reset_index(drop=True)[:k]
    summary = ' '.join(sortedDF.sort_values('Label')['Sentence'])

    return processedData, summary


def summarize(processedDirectory, destination_path, percentage=20, w_PS=1.5, w_LS=1, w_TF_IDF=2, w_NNP=1.25):
    """
    description
    """

    # Make a new file where we'll store all summaries
    if not os.path.exists(destination_path): 
        os.makedirs(destination_path)

    # For each file, do the summarization
    for file in os.listdir(processedDirectory):
        summary = []
        data = pd.read_csv(processedDirectory +  '/' + file)
        data['Words'] = data['Words'].apply(eval) # Opens list properly
        data = get_LS(data)
        data = get_PS(data)
        data = get_TF_IDF(data)
        data = find_propernouns(data)
        data,summary = get_summary(data, percentage, w_PS, w_LS, w_TF_IDF, w_NNP)

        fileName = os.path.basename(file) # also contains extension
        fileName = fileName.split('.',1)[0] # deletes extension
        output = open(destination_path + '/' + fileName + '.txt',"w+")
        output.write(summary)
        output.close()

    return data



##########


percentage = 20

paths = get_paths('cnn/stories')
stories = load_data(paths)
processedDirectory = preprocess(stories)
summarize(processedDirectory, r'cnn/summarized_stories')