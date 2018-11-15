import idf
import invertedindex as invi
import pickle
from nltk.corpus import reuters
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
import nltk
import time
import numpy as np
import evaluate as e

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
        invertedindex = pickle.load(open("invertedindex_test.pkl","rb"))
        l = []
        flag=0
        zero_result=0
        words_used=[]
        for i,j in pos_tag(word_tokenize(query.lower())):
            if j[0].lower() in ['a','n','v']:
                q = wnl.lemmatize(i,j[0].lower())
            else:
                q = wnl.lemmatize(i)
            if q not in invertedindex:
                words_used=[]
                zero_result=1
                break
            if q not in reuters.words("stopwords"):
                if flag==0:
                    l=list(invertedindex[q].keys())
                    words_used.append(q)
                else:
                    l1 = [value for value in l if value in list(invertedindex[q].keys())]
                    l=l1
                    if len(l)==0:
                        if added_vocab is not None:
                            words_used=[]
                            zero_result=1
                            break
                        else:
                            return l,words_used
                flag=1
        if added_vocab is not None:
            if zero_result or len(l)<5:
                for v in added_vocab:
                    try:
                        if v[1].isalpha() and len(list(invertedindex[v[1]].keys()))>0:
                            return list(invertedindex[v[1]].keys())+l,words_used+[v[1]]
                    except:
                        pass
            for v in added_vocab:
                if v[1].isalpha()==False or v[1] not in invertedindex.keys():
                    continue
                if len(l)>20:
                    l1 = [value for value in l if value in list(invertedindex[v[1]].keys())]
                    if len(l1)<20:
                        break
                    l=l1
                    words_used.append(v[1])
                else:
                    return l,words_used
            return l,words_used
        return l,words_used

def retrieve_results(docs, words_used):
    results=[]
    wnl = WordNetLemmatizer()
    tfidf = pickle.load(open("tfidf_test.pkl","rb"))
    q_list=[]
    for doc in docs:
        score=0
        for word in words_used:
            try:
                score+=tfidf[doc][word]
            except:
                pass
        results.append((score,doc))
    results = sorted(results,reverse=True)

    return [j for i,j in results]

    return [j for i,j in results]

def print_results(docs):
    for doc in docs[:20]:
        f=open("corpora/reuters/test/"+doc)
        print((f.readline()), end=' ')
        print((f.read()), end=' ')

def softmax(query1, query2):
    query1 = np.array(query1)
    query2 = np.array(query2)
    return np.dot(query1,query2)/(np.linalg.norm(query1)*np.linalg.norm(query2))

def query_expansion(query, model):
    tfidf = pickle.load(open("tfidf_test.pkl","rb"))

    wnl = WordNetLemmatizer()
    _, train_data = e.actual_results(open("corpora/reuters/cats.txt",'r'),False)
    tfidf_training = pickle.load(open("tfidf_training.pkl","rb"))
    q_list=[]
    docs=[]
    flag=0
    for i,j in pos_tag(word_tokenize(query.lower())):
        if j[0].lower() in ['a','n','v']:
            q = wnl.lemmatize(i,j[0].lower())
        else:
            q = wnl.lemmatize(i)
        if q not in reuters.words("stopwords"):
            q_list.append(q)
            if q in train_data.keys():
                temp=train_data[q]
            else:
                temp=train_data[i]
            if flag==0:
                docs=temp
                flag=1
            else:
                docs = [value for value in docs if value in temp]
    if len(q_list)==0:
        print("Enter something relevant")
        return None
    query_embedding = np.zeros(300)
    count=0
    for q in q_list:
        try:
            query_embedding+=model[q]
            count+=1
        except:
            pass
    if count>0:
        query_embedding/=count
    else:
        return None
    vocab=[]
    selected_vocab=[]
    for doc in docs:
        vocab+=tfidf_training[doc].keys()
    vocab=list(set(vocab))
    for i in vocab:
        try:
            selected_vocab.append((softmax(query_embedding,model[i]),i))
        except:
            pass
    selected_vocab = sorted(selected_vocab,reverse=True)
    return selected_vocab
