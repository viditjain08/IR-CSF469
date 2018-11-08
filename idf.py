from nltk import word_tokenize, pos_tag
from nltk.corpus import reuters
import nltk
from nltk.stem import WordNetLemmatizer
import collections
import os
import pickle
# nltk.download('reuters')
# nltk.download('wordnet')

def tokenize(sentence,vocab):
    wnl = WordNetLemmatizer()
    temp = pos_tag(word_tokenize(sentence.lower()))
    word_tokens=[]
    for i,j in temp:
        if j[0].lower() in ['a','n','v']:
            temp_i = wnl.lemmatize(i,j[0].lower())
        else:
            temp_i = wnl.lemmatize(i)
        if i not in reuters.words("stopwords"):
            if str(temp_i) not in word_tokens:
                vocab.append(str(temp_i))
            word_tokens.append(str(temp_i))
    return collections.Counter(word_tokens)


def get_idf():

    # print(tokenize(s))
    os.chdir(os.path.join(os.getcwd(),"corpora/reuters/training"))
    l = os.listdir(os.getcwd())
    vocab = list()
    for i in l:
        tokenize(open(i).read(),vocab)
        # print i
    with open("../../../idf.pkl", 'wb') as output:  # Overwrites any existing file.
        pickle.dump(collections.Counter(vocab), output)
    print("Inverse Document Frequency built")
