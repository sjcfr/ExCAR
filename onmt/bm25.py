import numpy as np
import pdb

from nltk.corpus import stopwords as pw

stopwords = set(pw.words("english"))

def tf(query_ls, doc_ls, avg_L, k=1.2, b=0.75):
    L = len(doc_ls)
    
    #query_set = set(query_ls)
    
    freq_terms = []
    for term in query_ls:
        
        freq_term = float(doc_ls.count(term)) / L
        freq_term = ((k + 1) * freq_term) / (k * (1 - b + b * float(L) / avg_L) + freq_term)
        #freq_terms[term] = freq_term
        freq_terms.append(freq_term)
    
    freq_terms = np.array(freq_terms)
    
    return freq_terms
    
    
def idf(query_ls, doc_term_freq, num_docs):
    
    query_set = query_ls
    
    idf_terms = []
    for term in query_set:
        try:
            term_freq = doc_term_freq[term]['freq']
        except:
            term_freq = 0
        idf_term = np.log((float(num_docs) - term_freq + 0.5) / (term_freq + 0.5))
        idf_terms.append(idf_term)
        
    idf_terms = np.array(idf_terms)
    
    return idf_terms
    
    
def rank(query_ls, candi_docs_ls, doc_term_freq, top_n=3, k=1.2, b=0.75, thre=9):
    
    scores = {}
    
    for doc_id, doc_ls in candi_docs_ls:
        tf_score = tf(query_ls, doc_ls, doc_term_freq['avg_L'], k, b)
        idf_score = idf(query_ls, doc_term_freq, doc_term_freq['num_docs']) 
        score = sum(tf_score * idf_score)
        
        scores[doc_id] = score
        
    sorted_scores = sorted(scores.items(), key=lambda x:x[1], reverse=True)[:top_n]
    ranked_doc_id = [int(i[0]) for i in sorted_scores if i[1] > thre]
    ranked_doc_score = [i[1] for i in sorted_scores if i[1] > thre]
    
    return ranked_doc_id, ranked_doc_score


def invidx(doc_dict):
    doc_term_freq = {}
    len_docs = []
    for ith, doc_ls in doc_dict.items():
        len_docs.append(len(doc_ls))
        for term in doc_ls:
            try:
                doc_term_freq[term]['freq'] += 1
                doc_term_freq[term]['doc_set'].add(ith)
            except:
                doc_term_freq[term] = {}
                doc_term_freq[term]['freq'] = 0
                doc_term_freq[term]['doc_set'] = set([ith])
    
    avg_L = np.mean(len_docs) 
    doc_term_freq['avg_L'] = avg_L
    doc_term_freq['num_docs'] = len(doc_dict.keys())          
    return doc_term_freq
    
    
def doc_match(query_ls, doc_term_freq):
    matched_doc_ls = []
    for term in query_ls:
        try:
            doc_term_freq[term]
            matched_doc_ls.extend(doc_term_freq[term]['doc_set'])
        except:
            pass
    
    matched_doc_ls = set(matched_doc_ls)
    
    return matched_doc_ls
    
            
        