import string
import pandas as pd
import json
import glob
import math
import re
import random
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import networkx
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import KElbowVisualizer

stop_words = set(stopwords.words('english'))
path = "CORD-19-research-challenge/**/*.json"


def Selecting_Documents(FilesPath,Percentage_of_files):
    files = glob.glob(FilesPath, recursive=True)
    count_of_files = math.floor(len(files) * Percentage_of_files)
    Extracted_Data = []
    for i in range(count_of_files):
        text_data =[]
        Extracted_Data.append(text_data)
        random_number = random.randrange(len(files))
        with open(files[random_number]) as file:
            data = json.load(file)
            for item in data['body_text']:
                text_data.append(item['text'])
    return Extracted_Data

def Nomrmalize_Text (text):
    Normalized_data =[]
    for i in range(len(text)):
        text = re.sub(r'[^a-zA-Z0-9\s]', '', str(text), re.I | re.A)
        text = text.lower()
        text = text.strip()
        text = "".join([char for char in text if char not in string.punctuation])
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in stop_words]
        tokens = '  '.join(tokens)
        Normalized_data.append(tokens)
    return Normalized_data

def Vectorizer (ListofDocuments):
    vectorizer = CountVectorizer (stop_words='english')
    matrix = vectorizer.fit_transform(ListofDocuments)
    # print(matrix.shape)
    return matrix

def KMeans_Clustering(matrix):
    clusters = 8
    model = KMeans(n_clusters=clusters, init='k-means++', n_init=10, max_iter=1000, tol=0.0001, random_state=42, n_jobs=-1)
    model.fit(matrix)
    model.labels_
    # print(silhouette_score(matrix,model.labels_))
    return model

def get_cluster_data(model,extracted_data):
    Cluster_data = []
    for i in range(len(model.cluster_centers_)):
        Temp_cluster_data = []
        for  j in range(len(model.labels_)):
            if i == model.labels_[j]:
                Temp_cluster_data.append(extracted_data[j])
        Cluster_data.append(Temp_cluster_data)
    # for  i in range(len(Cluster_data)):
    #     print(len((Cluster_data[i])))
    return Cluster_data

def Text_sentence_normanlization(Cluster_data):
    temp_data = str(Cluster_data)
    temp_data = re.sub(r'[^a-zA-Z0-9\s]', '', str(temp_data), re.I | re.A)
    temp_data = re.sub(r'\b(?:(?:https?|ftp)://)?\w[\w-](?:\.[\w-]+)+\S(?<![.,])', '', str(temp_data))
    temp_data = re.sub(r'\(.*?\)', '', str(temp_data))
    temp_data = re.sub(r'\[.*?\]', '', str(temp_data))
    temp_data = temp_data.lower()
    temp_data = temp_data.strip()
    tokenize_sentences = sent_tokenize(temp_data)
    return tokenize_sentences

def Text_Summarization(Cluster_wise_data ):
    for i in range(len(Cluster_wise_data)):
        print('cluster', i )
        tokenize_sentences = Text_sentence_normanlization(Cluster_wise_data[i])
        vectorizer = TfidfVectorizer(stop_words='english')
        matrix = vectorizer.fit_transform(tokenize_sentences)

        SimilatityMatrix = (matrix * matrix.T)
        SimilarityGraph = networkx.from_scipy_sparse_matrix(SimilatityMatrix)
        # networkx.draw_networkx(similarity_graph)
        DocScores = networkx.pagerank(SimilarityGraph)

        Sentence_Ranking = sorted(((score, index)for index, score in DocScores.items()),reverse=True)
        # print(ranked_sentences)
        Top_sentence_index = [Sentence_Ranking[index][1] for index in range(10)]
        Top_sentence_index.sort()
        # print(top_sentence_indices)

        path = 'Summarized_files'
        filename = 'cluster' + str(i)
        summarized_data = open(os.path.join(path, filename), 'w', encoding="utf-8")
        for index in Top_sentence_index:
            summarized_data.write(tokenize_sentences[index])
            summarized_data.write('\n')
        summarized_data.close()
    return 0

extracted_data = Selecting_Documents(path,.01)
norm_corpus= Nomrmalize_Text(extracted_data)
matrix = Vectorizer (norm_corpus)
model = KMeans_Clustering(matrix)
Cluster_wise_data = get_cluster_data(model,extracted_data)
Text_Summarization(Cluster_wise_data)



#-----------------------------------------------------------------------------------------------------------------------
# visualizer = KElbowVisualizer(model, k=(1,20))
# visualizer.fit(matrix)        # Fit the data to the visualizer
# # visualizer.show()
# k = visualizer.elbow_value_        # Finalize and render the figure
# print(k)


# def Selecting_Documents1(FilesPath,Percentage_of_files):
#     files = glob.glob(FilesPath, recursive=True)
#     count_of_files = math.floor(len(files) * Percentage_of_files)
#     #print(count_of_files)
#     dataframe = pd.DataFrame()
#     for i in range(count_of_files):
#         random_number = random.randrange(len(files))
#         with open(files[random_number]) as file:
#             data = json.load(file)
#             df = pd.json_normalize(data['body_text'])
#             dataframe = dataframe.append(df, ignore_index= True)
#     Extracted_Data =[]
#     for i in dataframe['text']:
#         Extracted_Data.append(i)
#     Extracted_Data = list(dict.fromkeys(Extracted_Data))
#     # print(len(Extracted_Data))
#     return Extracted_Data