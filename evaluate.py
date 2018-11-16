import collections
import get_results as gr

def actual_results(f,test=True):
    queries = []
    docs = dict()
    for line in f:
        x = line.split()
        if x[0][:8]=='training' and test==True:
            continue
        if x[0][:4]=='test' and test==False:
            continue
        category,docno = x[0].split('/')
        x.pop(0)
        for i in x:
            if i not in queries:
                queries.append(i)
                docs[i]=[docno]
            else:
                docs[i].append(docno)
    return queries, docs


def rprecision(ans_predicted, ans_actual, k):
    r_n = collections.Counter(ans_predicted[:k]) & collections.Counter(ans_actual)
    return len(r_n)

def normal_evaluation():
    with open("corpora/reuters/cats.txt") as f:
        queries,docs=actual_results(f)
        r_n5=0
        r_n10=0
        ave_p=float(0)
        for q in queries:
            doc_res,words_used = gr.simple_results(q)
            doc_res = gr.retrieve_results(doc_res,words_used)
            if doc_res!=-1 and doc_res!=[]:
                r_n5+=rprecision(doc_res,docs[q],5)
                r_n10+=rprecision(doc_res,docs[q],10)
                for i in range(1,12):
                    ave_p+=(float(rprecision(doc_res,docs[q],i))/float(i))
        stringy = ''
        stringy += str("P@5: ") + '\n'
        stringy += str(float(r_n5)/(5*len(queries))) + '\n'
        stringy += str("P@10: ") + '\n'
        stringy += str(float(r_n10)/(10*len(queries))) + '\n'
        stringy += str("MAP: ") + '\n'
        stringy += str(float(ave_p)/(11*len(queries))) + '\n'
        return stringy


def embedding_evaluation(model):
    with open("corpora/reuters/cats.txt") as f:
        queries,docs=actual_results(f)
        # model = gr.loadGloveModel("GloVe/glove.42B.300d.txt")
        r_n5=0
        r_n10=0
        ave_p=float(0)
        for q in queries:
            print(q)
            added_vocab = gr.query_expansion(q, model)
            if type(added_vocab)!=list or added_vocab==[]:
                added_vocab=None

            doc_res,words_used = gr.simple_results(q,added_vocab)
            doc_res = gr.retrieve_results(doc_res,words_used)

            if doc_res!=-1 and doc_res!=[]:
                r_n5+=rprecision(doc_res,docs[q],5)
                r_n10+=rprecision(doc_res,docs[q],10)
                for i in range(1,12):
                    ave_p+=(float(rprecision(doc_res,docs[q],i))/float(i))
        stringy = ''
        
        stringy += str("P@5: ") + '\n'
        stringy += str(float(r_n5)/(5*len(queries))) + '\n'
        stringy += str("P@10: ") + '\n'
        stringy += str(float(r_n10)/(10*len(queries))) + '\n'
        stringy += str("MAP: ") + '\n'
        stringy += str(float(ave_p)/(11*len(queries))) + '\n'
        return stringy
