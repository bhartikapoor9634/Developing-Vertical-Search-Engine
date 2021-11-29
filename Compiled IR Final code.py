#!/usr/bin/env python
# coding: utf-8

# # Importing necessary Libraries

# In[1]:


import requests
from bs4 import BeautifulSoup 
import csv
import pandas as pd
import os


# # Function designed for Crawler

# In[2]:


def mycrawler(seed, maxcount):
    seed_url= seed+ "/publications/"
    Q = [seed_url] 
    count = 0
    fills=[]  #list to append records
    
    baseurl = "https://pureportal.coventry.ac.uk"
    
    while(Q!=[] and count < maxcount):
        url = Q[0]   #accessing first element of queue
        Q = Q[1:]   #reassigning queue from second element #removing first element
        print("fetching " + url)
                 
        code = requests.get(url)   #Reading page
        plain = code.text            #HTML text of page
        #Passing HTML text to Beautiful Soup object to parse it and allow to search for any part of it
        s = BeautifulSoup(plain, "html.parser")  
        
        next_pg=s.find("a", class_="nextLink")
        if(next_pg!= None):
            print("next page")
            url_next_page=next_pg.get('href')
            
            #Normalisation
            if( url_next_page[0:7] != 'http://' and url_next_page[0:8]!='https://' ) :
                if(baseurl[len(baseurl) -1] == '/'):
                    url_next_page = baseurl + url_next_page 
                else:
                    url_next_page = baseurl + '/' + url_next_page
                        
            print("Link to next page is: ", url_next_page)
            
            #appending link to next page in queue
            Q.append(url_next_page) 
              
        #fetching all information about all divisions with class result-container
        results=s.find_all("div", class_="result-container")

        #retrieving details about paper and storing in dictionary
        for paper in results:
            fill={}
            paper_title = paper.find("h3", class_="title")
            paper_link = paper.find("a", class_="link")
            published=paper.find("span", class_="date")
            
            if(paper_title.text!= None):
                print("title of the paper is: ",paper_title.text)
                print("link to the paper is: ",paper_link.get('href'))
                print("Date of publication is:", published.text)
                print("autors' profile links:")

                author_name= []
                author_profile=[]
                for authors in paper.find_all("a", class_="link person"):

                    auth = authors.get('href')
                    auth_name=authors.string
                    author_name.append(auth_name)
                    author_profile.append(auth)
                    print(auth_name)
                    print(auth)
            
                fill['title']=paper_title.text
                fill['Link to paper']= paper_link.get('href')
                fill['Date of Publish']= published.text
                fill['author name']=author_name
                fill['authors profile link']=author_profile
                fills.append(fill)

            print("----------------------------------")

        #writing information to csv file
        filename = 'research.csv'
        with open(filename, 'w', newline='', encoding="utf-8") as f:
            w = csv.DictWriter(f,['title','Link to paper','Date of Publish','author name','authors profile link'])
            w.writeheader()
            for fill in fills:
                w.writerow(fill)
        count=count+1
        print("Crawler ran successfully and results stored in Research.csv file ") 
        


# # Running Crawler Manually

# In[3]:


run_crawler=input("Do you want to run a crawler (y/n): ")
if (run_crawler.lower()=='y'):
    mycrawler('https://pureportal.coventry.ac.uk/en/organisations/school-of-life-sciences',20)

elif(run_crawler.lower()=='n'):
    if(os.path.isfile('research.csv')):
        print("Search Engine initiated")
    else: 
        print("No Crawler output exists, you need to run crawler first!")
else:
    print("Invalid input received. Please enter your results in y/n")


# # Storing contents of csv file into dataframe

# In[4]:


#Reading csv file and storing contents in dataframe
df=pd.read_csv("research.csv", names=['title','Link to paper','Date of Publish','author name','authors profile link'],
               encoding= 'unicode_escape')
#Dropping first row containing labels only
df=df.iloc[1:,:]
print("No. of research papers retrieved: ", df.title.count())
df.head()


# # Dropping papers whose authors are not from CU

# In[5]:


#there are some retrieved papers whose authors are not part of CU anymore
df.loc[df['author name'] == '[]']


# In[6]:


#Removing papers whose authors are Not part of CU
df.drop(df.loc[df['author name']=="[]"].index, inplace=True)
print("No. of research papers left after removing papers fron non CU authors: ", df.title.count())
df.head()


# In[7]:


df.info()


# # Storing title of paper in list

# In[8]:


#Storing title of research paper in list
title_list = df["title"].to_list()
print(type(title_list))


# In[9]:


#Displaying list of titles of research paper
title_list


# # Preprocessing title of paper

# In[10]:


#Data pre processing using Tokenizer, Stemmer and Stop Word removal process
#importing necessary libraries
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')

stemmer = PorterStemmer()
def preprocessing(doc):
    # changing sentence to lower case
    doc = doc.lower()
    print(doc)
    tokenizer = nltk.RegexpTokenizer(r"\w+")   #removing punctuation
     # tokenize into words
    words = tokenizer.tokenize(doc)         
    print(words)
    # removing stop words
    words = [word for word in words if word not in stopwords.words("english")]
    print(words)
    # stemming
    words = [stemmer.stem(word) for word in words if word.isalpha()]
    print(words)
    # join words to make sentence
    doc = " ".join(words)
    return doc


# In[11]:


#Pre-processing the title list
documents = [preprocessing(document) for document in title_list]
print(documents)


# In[12]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
tfidf_model = vectorizer.fit_transform(documents)
print(tfidf_model)


# In[13]:


print(tfidf_model.shape)


# In[14]:


# print the full sparse matrix
sparse=tfidf_model.toarray()
sparse


# In[15]:


title_vec=pd.DataFrame(tfidf_model.toarray(), columns = vectorizer.get_feature_names())


# In[16]:


title_vec


# # Inverted Index

# In[17]:


#Each word feature will be mapped against the doc ids where it is found.
#Search Pattern will use this indexed corpus for faster search.

from collections import defaultdict
from nltk.tokenize import sent_tokenize, word_tokenize

#Index is a map of word in documents where it is found in
inverted_index = defaultdict(set)
from nltk.corpus import stopwords
from nltk.stem import snowball
stwords = set(stopwords.words('english'))

# Maintain the reference to the document by its index in the corpus list
for docid, c in enumerate(title_list):
    #print(docid , c)
    for sent in sent_tokenize(c):
        for word in word_tokenize(sent):
            if (word.isalpha()):
                word_lower = word.lower() 
            if word_lower not in stwords:
                stemmer = snowball.SnowballStemmer('english')
                word_stem = stemmer.stem(word_lower)                
              
            # indexes will be incremented when it will have any new word rather than building from scratch
                if (word_stem in inverted_index) and (docid not in inverted_index[word_stem]):   
                    inverted_index[word_stem].add(docid)
                 # New term in inverted_index
                else:
                    inverted_index[word_stem].add(docid)   

print(len(inverted_index.keys()))
print(inverted_index)


# # sorting inverted index

# In[18]:


#Sorting inverted index
sort_inverted_index=sorted(inverted_index.items())
sorted_inverted_index = defaultdict(set)
sorted_inverted_index= dict(sort_inverted_index)
print(sorted_inverted_index)


# In[19]:


#Displaying Postings list from inverted index
def post_list(term):
    return sorted_inverted_index[term]
print(post_list("ambient"))


# # Query Processor

# In[20]:


#Here the search patterns are searched in OR condition.
def preprocess_and_search(query):
    matched_documents = set()      #to store the information about papers with which the query matches
    #Pre-Processing user query
    for word in word_tokenize(query):
        
        if word.isalpha():
            word_lower = word.lower() 
            
            if (word_lower not in stwords):
                word_stem = stemmer.stem(word_lower)
                query_stem.append(word_stem)
                
                # fetching docid of documents with which query matches using inverted index
                matches = sorted_inverted_index.get(word_stem)
                
                if matches:
                    # The operator |= is a short hand for set union
                    #Gathering all the matching docids in a set 
                    matched_documents |= matches      #union signifies OR operation
                    print("Documents found for ", word_stem, "are : ",matched_documents)
    
    print("User query after pre-processing ",query_stem)
    return matched_documents


# In[26]:


# the below search function will return the doc id which is aligned to the index of the corpus.
query_stem = []
searchstring=input("enter your search string ")
doc_id = preprocess_and_search(searchstring)
print("Final matching list : ", doc_id)
print("No of matching documents found: ",len(doc_id))


# In[27]:


#extracting tfid scores for only matched docids and features 
tf_idf_rank_list=list()
if(doc_id!= set()):
    for i in (doc_id):
        tf_idf_rank=0
        for j in range(len(query_stem)):
            try:
                q=query_stem[j]
                tf_idf_rank += title_vec.iloc[i][q]
            except KeyError:
                tf_idf_rank+=0
        tf_idf_rank_list.insert(i,tf_idf_rank)
    print(tf_idf_rank_list)
    print(len(tf_idf_rank_list))


# In[28]:


sorted_tf_idf_rank_list=(sorted( [(x,i) for (i,x) in enumerate(tf_idf_rank_list)], reverse=True ))
print(sorted_tf_idf_rank_list)


# In[29]:


#Papers fetched with relevancy
if (doc_id!= set()):
    for i in range(len(sorted_tf_idf_rank_list)):
        x=sorted_tf_idf_rank_list[i][1]
        y=list(doc_id)[x]
        
        print("Title of Paper:",df.iloc[y,0])
        print("Link of Paper:",df.iloc[y,1])
        print("Date of Paper Published:",df.iloc[y,2])
        print("Author names:",df.iloc[y,3])
        print("Link to Author Profiles:",df.iloc[y,4])
        print("---------------------------------------")

else:
    print("No match found")


# In[30]:


#papers fetched without relevancy
if (doc_id!= set()):
    for i in doc_id:
        print("Title of Paper:",df.iloc[i,0])
        print("Link of Paper:",df.iloc[i,1])
        print("Date of Paper Published:",df.iloc[i,2])
        print("Author names:",df.iloc[i,3])
        print("Link to Author Profiles:",df.iloc[i,4])
        print("---------------------------------------")
else:
    print("No match found")


# # Task 2

# # Installing Feedparser package to read RSS feed

# In[ ]:


pip install feedparser


# # Gathering Data

# In[32]:


#Fetching news using RSS feed
#getting summary instead of full RSS feed
import feedparser
lst=[]
Y=[]
NewsFeed_business = feedparser.parse("http://feeds.bbci.co.uk/news/business/rss.xml")
NewsFeed_sci = feedparser.parse("http://feeds.bbci.co.uk/news/science_and_environment/rss.xml")
NewsFeed_arts = feedparser.parse("http://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml")
NewsFeed_tech = feedparser.parse("http://feeds.bbci.co.uk/news/technology/rss.xml")
NewsFeed_pol= feedparser.parse("http://feeds.bbci.co.uk/news/politics/rss.xml")

for entry in NewsFeed_business.entries:
    x= entry.summary
    y=0
    lst.append(x)
    Y.append(y)

for entry in NewsFeed_sci.entries:
    y= entry.summary
    lst.append(y)
    y=1
    Y.append(y)

for entry in NewsFeed_arts.entries:
    z= entry.summary
    lst.append(z)
    y=2
    Y.append(y)

for entry in NewsFeed_tech.entries:
    p= entry.summary
    lst.append(p)
    y=3
    Y.append(y)

for entry in NewsFeed_pol.entries:
    p= entry.summary
    lst.append(p)
    y=4
    Y.append(y)

print("no of news headlines fetched: ",len(lst))
print("number of labels is: ", len(Y))
print(lst)


# # Pre- processing

# In[33]:


#Pre-processing the fetched feed

filtered_docs = []
for doc in lst:
    tokens = word_tokenize(doc)
    tmp = ""
    for w in tokens:
        if w not in stwords:
            if(w.isalpha()):
                stemmer = snowball.SnowballStemmer('english')
                tmp += stemmer.stem(w) + " "
    filtered_docs.append(tmp)

print(filtered_docs)


# # Vectorisation

# In[34]:


#TfidfVectorisation
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(filtered_docs)
print(X.todense())
words = vectorizer.get_feature_names()
print(words)


# # Using Elbow method to predict number of clusters

# In[35]:


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
wcss = []
for i in range(1,7):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=100,n_init=15,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,7),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.savefig('elbow.png')
plt.show()


# # Designing Cluster Model

# # K=5

# In[36]:


#Clustering with K=5
import numpy as np
from sklearn.cluster import KMeans
# trying to make 5 clusters as there are five categories: Business, Science, art, technology and politics
K = 5
model_K5 = KMeans(n_clusters=K)
model_K5.fit(X)

print("cluster no. of input documents, in the order they received:")
print(model_K5.labels_)


# We look at 5 clusters generated by k-means.
common_words = model_K5.cluster_centers_.argsort()[:,-1:-26:-1]
for num, centroid in enumerate(common_words):
    print(str(num) + ' : ' + ', '.join(words[word] for word in centroid))


# # K=4

# In[37]:


#Clustering with K=4
import numpy as np
from sklearn.cluster import KMeans
# trying to make 4 clusters 
K = 4
model_K4 = KMeans(n_clusters=K)
model_K4.fit(X)

print("cluster no. of input documents, in the order they received:")
print(model_K4.labels_)


# We look at 4 clusters generated by k-means.
common_words = model_K4.cluster_centers_.argsort()[:,-1:-26:-1]
for num, centroid in enumerate(common_words):
    print(str(num) + ' : ' + ', '.join(words[word] for word in centroid))


# # K=3

# In[38]:


#Clustering with k=3
import numpy as np
from sklearn.cluster import KMeans
# trying to make 3 clusters 
K = 3
model_K3 = KMeans(n_clusters=K)
model_K3.fit(X)

print("cluster no. of input documents, in the order they received:")
print(model_K3.labels_)


# We look at 3 clusters generated by k-means.
common_words = model_K3.cluster_centers_.argsort()[:,-1:-26:-1]
for num, centroid in enumerate(common_words):
    print(str(num) + ' : ' + ', '.join(words[word] for word in centroid))


# # Predicting Clusters

# In[39]:


#testing document
#trying with different categories of news to check what clusters we get
#Business, Science, art, technology, Politics
#these testing document contains queries containing stop words, Punctuations, mixture of Upper case and lower case letters
test_doc = ['The UK energy market has seen 24 companies collapse since September after wholesale gas prices rose',
            'Sir Andrew Pollard warns against possible moves to reverse planned investment in science',
            'Adele album became 2021 fastest seller',
            'Huge fines on makers of insecure smart devices.', 
            'PM under pressure has to act on migrant sea crossings.'
           ]

#Preprocessing on test document
filtered_test_docs = []
for doc in test_doc:
    tokens = word_tokenize(doc)
    tmp = ""
    for w in tokens:
        if w not in stwords:
            if(w.isalpha()):
                tmp += stemmer.stem(w) + " "
    filtered_test_docs.append(tmp)

print("User query after preprocessing is: ",filtered_test_docs)

for i in range (len(filtered_test_docs)):
#Predicting cluster
    V = vectorizer.transform([filtered_test_docs[i]])
    prediction_K5 = model_K5.predict(V)
    prediction_K4 = model_K4.predict(V)
    prediction_K3 = model_K3.predict(V)
    print("The cluster with K=5 is: ", prediction_K5, "The cluster with K=4 is: ", prediction_K4, "The cluster with K=3 is: ", prediction_K3)


# In[40]:


#printing assigned labels
print(Y)


# In[41]:


#printing predicted labels by model with k=5 clusters
print(model_K5.labels_)


# In[42]:


from sklearn.metrics import classification_report
y_true = Y
y_pred = model_K5.labels_
target_names = ['Business', 'Science', 'art', 'technology', 'Politics']
print(classification_report(y_true, y_pred, target_names=target_names))

