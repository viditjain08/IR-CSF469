import idf
import invertedindex as invi
import pickle
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
import get_results as gr

if __name__ == "__main__":
    option=0
    wnl = WordNetLemmatizer()
    flag=0
    while option!=4:
        print("Choose among the following")
        print("1. Make the dict")
        print("2. Search a query")
        print("3. Make a modified query with embeddings")
        print("4. Exit")
        option=int(input())
        if option==4:
            break
        if option==1:
            print("This may take between 1-2 hours")
            idf.get_idf()
            invi.get_inverted_tfidf()
        elif option==2:
            print("Enter query")
            query=str(input())
            if query is None or len(query)==0:
                print("Query cannot be empty")
                continue
            docs = gr.simple_results(query)
            if docs==-1:
                print("Some error occured")
                continue
            if docs==[]:
                print("No results found")
                continue
            gr.print_results(docs,query)
        else:
            if flag==0:
                print("May take 5 minutes")
                model = gr.loadGloveModel("GloVe/glove.42B.300d.txt")
            flag=1
            print("Enter query")
            query=str(input())
            if query is None or len(query)==0:
                print("Query cannot be empty")
                continue
            added_vocab = gr.query_expansion(query, model)
            docs = gr.simple_results(query,added_vocab[:2])
            if docs==-1:
                print("Some error occured")
                continue
            if docs==[]:
                print("No results found")
                continue
            gr.print_results(docs,query)
