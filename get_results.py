import idf
import invertedindex as invi
import pickle
from nltk.corpus import reuters
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
import nltk
import time
import numpy as np

# def loadGloveModel(gloveFile):
#     print("Loading Glove Model")
#     f = open(gloveFile,'r')
#     model = []
#     embedding = []
#     for line in f:
#         splitLine = line.split()
#         word = splitLine[0]
#         model.append(word)
#         embedding.append(np.array([float(val) for val in splitLine[1:]]))
#     print("Done."),
#     print(len(model)),
#     print(" words loaded!")
#     return model,embedding

def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    model = dict()
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        model[word] = np.array([float(val) for val in splitLine[1:]])
    print("Done."),
    print(len(model)),
    print(" words loaded!")
    return model

def simple_results(query,added_vocab=None):
        wnl = WordNetLemmatizer()
        invertedindex = pickle.load(open("invertedindex.pkl","rb"))
        l = []
        flag=0

        for i,j in pos_tag(word_tokenize(query.lower())):
            if j[0].lower() in ['a','n','v']:
                q = wnl.lemmatize(i,j[0].lower())
            else:
                q = wnl.lemmatize(i)
            if q not in invertedindex:
                return []
            if q not in reuters.words("stopwords"):
                if flag==0:
                    l=list(invertedindex[q].keys())
                else:
                    l1 = [value for value in l if value in list(invertedindex[q].keys())]
                    l=l1
                    if len(l)==0:
                        return []
                flag=1
        if added_vocab is not None:
            for v in added_vocab:
                if len(l)>30:
                    print(v[1])
                    l1 = [value for value in l if value in list(invertedindex[v[1]].keys())]
                    l=l1
                else:
                    break
        return l


def print_results(docs, query):
    results=[]
    wnl = WordNetLemmatizer()
    tfidf = pickle.load(open("tfidf.pkl","rb"))
    for doc in docs:
        score=0
        for i,j in pos_tag(word_tokenize(query.lower())):
            if j[0].lower() in ['a','n','v']:
                q = wnl.lemmatize(i,j[0].lower())
            else:
                q = wnl.lemmatize(i)
            if q not in reuters.words("stopwords"):
                score+=tfidf[doc][q]
        results.append((score,doc))
    results = sorted(results,reverse=True)[:20]
    for score,doc in results:
        f=open("corpora/reuters/training/"+doc)
        print((f.readline()), end=' ')
        print(("Score: "),score)
        print((f.read()), end=' ')

def softmax(query1, query2):
    query1 = np.array(query1)
    query2 = np.array(query2)
    return np.dot(query1,query2)/(np.linalg.norm(query1)*np.linalg.norm(query2))

def query_expansion(query, model):
    a = time.time()
    tfidf = pickle.load(open("tfidf.pkl","rb"))

    wnl = WordNetLemmatizer()
    docs=simple_results(query)
    q_list=[]
    for i,j in pos_tag(word_tokenize(query.lower())):
        if j[0].lower() in ['a','n','v']:
            q = wnl.lemmatize(i,j[0].lower())
        else:
            q = wnl.lemmatize(i)
        if q not in reuters.words("stopwords"):
            q_list.append(q)
    if len(q_list)==0:
        print("Enter something relevant")
        return -1
    query_embedding = np.zeros(300)
    count=0
    for q in q_list:
        try:
            query_embedding+=model[q]
            count+=1
        except:
            pass
    try:
        query_embedding/=count
    except:
        return -1
    print(count)
    vocab=[]

    selected_vocab=[]
    for doc in docs[:50]:
        vocab+=tfidf[doc].keys()
    vocab=list(set(vocab))
    for i in vocab:
        try:
            selected_vocab.append((softmax(query_embedding,model[i]),i))
        except:
            pass
    selected_vocab = sorted(selected_vocab,reverse=True)
    print(time.time()-a),"sec"
    return selected_vocab
