import idf
import invertedindex as invi

if __name__ == "__main__":
    print("Choose among the following")
    print("1. Make the dict")
    print("2. Search a query")
    print("3. Exit")
    option=int(raw_input())
    while option!=3:
        if option==1:
            print("This may take between 1-2 hours")
            idf.get_idf()
            invi.get_inverted_tfidf()
        else:
            pass
        option=int(raw_input())
