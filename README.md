# COVID 19 Text Data Summarization with NLP

![head](https://github.com/PurushothamVadde/COVID_19_Text_Data_Summarization_with_NLP/blob/master/Text-Summarization.jpg)

## Packages Required for Project:
- nltk
- glob
- re
- sklearn
- networkx
- json
- yellowbrick

In this project i am taking text files of json format related to **COVID** and performing the KMeans clustering on documents so that the similar type of documents are come under one cluster, after clustering the documents i am taking the documents from each cluster and performing the text summarization by using page rank algorithm technique.

### Json files 
We are taking the Json files as input document files, in the json file we have the dictionary object called body_text with value as array, the array has keys by name of text and the value as **text data**, we take the all text values and perform summarization techniques on that text data. 

The projects have below files:
## summarizer.py
The summarizer file contains the below functions
### Selecting_Documents(FilesPath,Percentage_of_files):

In this function i am taking the input files path and percentage of files as input arguments, we use the glob function to read the all  files in the folders, and the we get the count of number of files that we need to process and stored in the count_of_files variable.

>  files = glob.glob(FilesPath, recursive=True) \
>  count_of_files = math.floor(len(files) * Percentage_of_files)

we iterate through the len of count_of_files and select the random files by using below code
>random_number = random.randrange(len(files)) 

By using the json package we open the each randomly selected json file and read the data where the key is 'text', we get the values where the key is text and stored into the text_data for a file with below code.
> data = json.load(file) \
> for item in data['body_text']: \
>     text_data.append(item['text']) 

After we append that text_data to Extracted_data list for each files, by this we get the list of with each element as 1 document data.

### Nomrmalize_Text (text):

The Extracted_text list with each element contains the text from each document is passed as input argument to the Nomrmalize_Text() function, we will iterate throught the each element in the list perform the text cleaning by removing the stopwords, punctuations, and we also remove the text other than numbers, and alphabets by using below regular expression.
> text = re.sub(r'[^a-zA-Z0-9\s]', '', str(text), re.I | re.A) 

After cleaning the text we perfornm the word tokenize on the above text data.
> tokens = word_tokenize(text) 

After performing the word tokenize we join each word and append the string to a list so that we will get a list with each element as normalized text data for each document.

### Vectorizer (ListofDocuments):

The Normalized textdata list from the above function is passed as input argument to the Vectorizer( ) , in this we perform the vectorizer by using **CountVectorizer**
> vectorizer = CountVectorizer (stop_words='english') \
> matrix = vectorizer.fit_transform(ListofDocuments)

After performing the countVectorizer we will get a matrix of size number_of_documents * number of features.

### KMeans_Clustering(matrix):
After performing the CountVecorizer we get the matrix for the complete list of documents with each row represents the document, the matrix is passed to the KMeans model using fit function,  and we pass the n_clusters as a input argument to the KMeans function so that based on the n_clusters value the clusters will be formed.
> model = KMeans(n_clusters=clusters, init='k-means++', n_init=10, max_iter=1000, tol=0.0001, random_state=42, n_jobs=-1)  
> model.fit(matrix)

- finding the K Value: \
To find the number of clusters,  I used the kelbow function from the yellowbricks package, by passing the model and k vlaue range as input argument to the KElbowVisualizer(), we will get the optimized k value .

> visualizer = KElbowVisualizer(model, k=(1,20))
> visualizer.fit(matrix)    
> visualizer.show()

In the below plot we can see that the optimal K Value is 8. \
i got the optimal kvalue as 8 for 1% of data from the covid json files.


  ![KValue](https://github.com/PurushothamVadde/cs5293sp20-project2/blob/master/Kvalue.png) 
    
### get_cluster_data(model,extracted_data):
By using the get_cluster_data() we get the documents related to the each cluster by using model.lables_, the model.lables_ contain the labels of documents in each cluster. \
the get_cluster_data() takes the input arguments as extracted data and model, by using model.cluster_centers_ we iterate through each cluster and in each cluster using model.lables_ we iterate through each document and we append the data for each document in the cluster so that we will get a list with each element in the list represent the cluster data. \

> for i in range(len(model.cluster_centers_)):
>>  Temp_cluster_data = [] \
>>  for  j in range(len(model.labels_)):
>>>    if i == model.labels_[j]:
>>>>      Temp_cluster_data.append(extracted_data[j])
>>  Cluster_data.append(Temp_cluster_data)


### Text_Summarization(Cluster_wise_data):

we pass the cluster wise data from the above function to the  Text_Summarization(Cluster_wise_data) function, we iterate through the each cluster wise data and we perform the below operations to summarize the text of each cluster.

- Tokenize the sentences:
We perform the sentence tokenize on by  using the nltk.sent_tokenize , by this we will get the tokenized data for each cluster data.
> tokenize_sentences = Text_sentence_normanlization(Cluster_wise_data[i])

-Vectorize the tokenize sentences:
we apply the TfidfVectorizer function on the tokenize sentences to convert the data into machine understanding format, after transform the data we will get a matrix.
> vectorizer = TfidfVectorizer(stop_words='english')
> matrix = vectorizer.fit_transform(tokenize_sentences)

- Generate similarity matrix:
we get the similarity matrix by multiplying the matrix with transpose of matrix.
> SimilatityMatrix = (matrix * matrix.T)

- Pagerank Algorithm:
The similary matrix is passed to the sparse matrix function in networkx package we ge a similarity graph, the similarity graph is passed through the pagerank alogirith so that we get the scores of the each sentence.
> SimilarityGraph = networkx.from_scipy_sparse_matrix(SimilatityMatrix)
> DocScores = networkx.pagerank(SimilarityGraph)

- Reading the top Sentences
Based on the pagerank score we sort the sentences and with high ranking and we select the top 10 sentences and we write to a document.


## Assumptions:

- i implemented the summarizer project for only **1 %** of data from the below covid dataset:
https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge

- Forigen languages are not handled.








