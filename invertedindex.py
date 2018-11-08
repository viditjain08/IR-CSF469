from nltk import word_tokenize, pos_tag
from nltk.corpus import reuters
import nltk
from nltk.stem import WordNetLemmatizer
import collections
import os
import pickle
from math import log10

# nltk.download('reuters')
# nltk.download('wordnet')

def tokenize(sentence):
    wnl = WordNetLemmatizer()
    temp = pos_tag(word_tokenize(sentence.lower()))
    word_tokens=[]
    for i,j in temp:
        if j[0].lower() in ['a','n','v']:
            temp_i = wnl.lemmatize(i,j[0].lower())
        else:
            temp_i = wnl.lemmatize(i)
        if i not in reuters.words("stopwords"):
            word_tokens.append(str(temp_i))
    return collections.Counter(word_tokens)

def tfidf():
    vocab = pickle.load(open("idf.pkl"))
    os.chdir(os.path.join(os.getcwd(),"corpora/reuters/training"))
    l = os.listdir(os.getcwd())
    invertedindex={}
    tf={}
    count=0
    print "Total Documents", len(l)
    for i in l:
        length=0
        temp_tokens = tokenize(open(i).read())
        tf[i]={}
        for k,v in temp_tokens.items():
            if k not in invertedindex.keys():
                invertedindex[k] = {i:v}
            else:
                invertedindex[k][i]=v
            temp_tfidf=(1+log10(float(v)))*log10(len(l)/vocab[k])
            tf[i][k]=temp_tfidf
            length+=temp_tfidf**2
        tf[i]["length"]=length
        count+=1
        if count%1000==0:
            print count, "documents indexed"
        # print invertedindex
        # print tf
    return invertedindex,tf
def get_inverted_tfidf():

    # print(tokenize(s))
    invertedindex,tf = tfidf()
    with open("../../../invertedindex.pkl", 'wb') as output:  # Overwrites any existing file.
        pickle.dump(invertedindex, output)
    with open("../../../tfidf.pkl", 'wb') as output:  # Overwrites any existing file.
        pickle.dump(tf, output)
