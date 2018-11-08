import idf
import invertedindex as invi
import pickle
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer

if __name__ == "__main__":
    option=0
    wnl = WordNetLemmatizer()
    while option!=3:
        print("Choose among the following")
        print("1. Make the dict")
        print("2. Search a query")
        print("3. Exit")
        option=int(raw_input())
        if option==3:
            break
        if option==1:
            print("This may take between 1-2 hours")
            idf.get_idf()
            invi.get_inverted_tfidf()
        else:
            print("Enter query")
            query=str(raw_input())
            if query is None or len(query)==0:
                print("Query cannot be empty")
                continue
            try:
                invertedindex = pickle.load(open("invertedindex.pkl","rb"))
                tfidf = pickle.load(open("tfidf.pkl","rb"))
                l = []
                flag=0
                for i,j in pos_tag(word_tokenize(query.lower())):
                    if j[0].lower() in ['a','n','v']:
                        q = wnl.lemmatize(i,j[0].lower())
                    else:
                        q = wnl.lemmatize(i)
                    if q not in invertedindex:
                        print("No results found")
                        flag=-1
                        break
                    if flag==0:
                        l=invertedindex[q].keys()
                    else:
                        l1 = [value for value in l if value in invertedindex[q].keys()]
                        l=l1
                        if len(l)==0:
                            print("No results found")
                            continue
                    flag=1
                docs=[]
                if flag==-1:
                    continue
                for doc in l:
                    score=0
                    for i,j in pos_tag(word_tokenize(query.lower())):
                        if j[0].lower() in ['a','n','v']:
                            q = wnl.lemmatize(i,j[0].lower())
                        else:
                            q = wnl.lemmatize(i)
                        score+=tfidf[doc][q]
                    docs.append((score,doc))
                results = sorted(docs,reverse=True)[:20]
                for score,doc in results:
                    f=open("corpora/reuters/training/"+doc)
                    print(f.readline()),
                    print("Score: "),score
                    print(f.read()),
            except:
                print("Dict not built/ Some error")
