import random
import numpy as np
import copy
import pdb

from onmt.bm25 import * 


def rank(query_ls, doc_ls):
    len_overlap_ls = []
    for doc in doc_ls:
        #pdb.set_trace()
        len_overlap_ls.append([len(set(doc[1]).intersection(query_ls)), doc])
        
    sorted_scores = sorted(len_overlap_ls, key=lambda x:x[0], reverse=True)[:3]

    ranked_doc_id = [int(i[0]) for i in sorted_scores]
    ranked_doc_score = [i[1] for i in sorted_scores]
    #pdb.set_trace()
    return ranked_doc_id, ranked_doc_score
        
    

class Event():
    def __init__(self, idx):
        self.idx = idx
        self.text = None
        self.effects = set()
        self.causes = set()
        self.words = []
        

class AdvExample():
    def __init__(self, typ, ask_for, hypo_id, alt_id, hypo=None, alt=None, ans=None, num_hop=None):
        self.typ = typ
        self.ask_for = ask_for
        self.hypo_id = hypo_id
        self.alt_id = alt_id
        self.hypo = hypo
        self.alt = alt
        self.ans = ans
        self.scores = []
        self.num_hop = num_hop
        
    def priority_cal(self, thre=1,  by_prob=False):
        
        scores = np.array(self.scores)
        #delta_scores = scores - scores[:,self.ans].reshape(scores.shape[0], 1)
        
        #pdb.set_trace()
        delta_scores =  scores[:,self.ans].reshape(scores.shape[0], 1) - scores 
        self.delta_less_than_0 = ((delta_scores < 0).sum(0) > thre)
        self.avg_score = delta_scores.mean(0)
        #self.avg_score[0] = max(max(self.avg_score) + 1, 1)
        
        #avg_score_alts = (self.avg_score * self.delta_less_than_0).tolist()
        #avg_score_alts = (np.abs(self.avg_score) * (self.delta_less_than_0 * 2 - 1)).tolist()
        avg_score_alts = (np.abs(self.avg_score) * (-(self.delta_less_than_0 * 2 - 1))).tolist()
        
        avg_score_alts.pop(self.ans)
        avg_score_alts = np.array(avg_score_alts)
        
        #if self.typ == 0 and all(avg_score_alts >= 0):
        avg_score_alts = avg_score_alts - min(avg_score_alts) - 1e-6
            
        alt_id = copy.deepcopy(self.alt_id)
        alt_id.pop(self.ans)  
        if not by_prob: 
            priorities = {}
            for a, priority in zip(alt_id, avg_score_alts):
                priorities[(self.hypo_id, (self.alt_id[self.ans], a), self.ask_for, self.typ, self.num_hop)] = priority
                
            keep_dict = {}
            candi_dict = {}
            
            '''
            for k in priorities.keys():
                if priorities[k] < 0:
                    keep_dict[k] = priorities[k]
                else:
                    candi_dict[k] = priorities[k]
            '''
                    
            min_priority = 0
            
            for k in priorities.keys():
                if priorities[k] < 0:
                    if priorities[k] < min_priority: 
                        keep_dict = {k: priorities[k]}
                        min_priority = priorities[k]
                    else:
                        candi_dict[k] = priorities[k]
                else:
                    candi_dict[k] = priorities[k]
                    
            #pdb.set_trace()
        else:
            probs = np.array(copy.deepcopy(avg_score_alts))
            probs = (probs - probs.min()) / (probs.max() - probs.min())
            probs = 1 - probs
            probs = probs / sum(probs)
            
            keep_id = np.random.choice([i for i in range(len(alt_id))], probs)
            
            keep_dict = {(self.hypo_id, (self.alt_id[self.ans], alt_id[keep_id]), self.ask_for, self.typ): avg_score_alts[keep_id]}
            candi_dict = {}
            
            for a, priority in zip(alt_id, avg_score_alts):
                if a != keep_id:
                    candi_dict[(self.hypo_id, (self.alt_id[self.ans], a), self.ask_for, self.typ)] = priority
        
        return keep_dict


class Example(object):
    
    def __init__(self,
                 sample_id,
                 hypo_id,
                 alt_id,
                 ask_for,
                 typ,
                 num_hop,
                 random_ans=True,
                 ans=None
                 ):
        self.sample_id = sample_id
        self.alt_id = alt_id
        self.alt = []
        self.hypo_id = hypo_id
        self.hypo = None
        self.ask_for = ask_for
        self.typ = typ
        self.num_hop = num_hop
        self.hypo_anchor_id = []
        self.hypo_anchor = []
        self.alt_anchor_id = []
        self.alt_anchor = []
        self.evidence_id = []
        self.evidence = []
        self.adjacency_matrix = []
        
        if random_ans:
            self.ans = random.randint(0, 1)
            if self.ans == 1:
                self.alt_id = [self.alt_id[1], self.alt_id[0]]
        else:
            self.ans = ans
        
    def attr_matching(self, events_dict):
        self.hypo = events_dict[str(self.hypo_id)].text
        self.alt = [events_dict[str(self.alt_id[0])].text, events_dict[str(self.alt_id[1])].text]
            
    def anchor_matching(self, events, doc_term_freq, top_n=3, thre=9, stopwords=None):
        
        def anchor_rank(event_id, doc_term_freq, events, top_n=top_n, thre=thre):
            query_ls = events[str(event_id)].words
            query_ls = list(set(query_ls) - stopwords) # !!!
            
            matched_doc_id_ls = doc_match(query_ls, doc_term_freq)
            matched_doc_ls = []
            
            for doc_id in matched_doc_id_ls:
                matched_doc_ls.append([doc_id, events[str(doc_id)].words])
                
            query_ls_cp = []
            for word in query_ls:
                word = word.lower()
                word = word.replace('.','')
                query_ls_cp.append(word)
            
            query_ls_cp = set(query_ls_cp)
            query_ls_cp = list(query_ls_cp - stopwords)
            ranked_doc_ls, ranked_doc_score = rank(query_ls_cp, matched_doc_ls)
            
            return ranked_doc_ls, ranked_doc_score
        
        ranked_doc_scores = [] 
        
        for a_id in self.alt_id:
            
            ranked_anchor_ls, ranked_doc_score_ls = anchor_rank(a_id, doc_term_freq, events, top_n, thre)
            
            ranked_doc_scores.extend(ranked_doc_score_ls)
            
            self.alt_anchor_id.append(ranked_anchor_ls)
            
            self.alt_anchor.append([])
            for ranked_anchor in ranked_anchor_ls:                
                self.alt_anchor[-1].append(events[str(ranked_anchor)].text)
                    
        self.hypo_anchor_id, _ = anchor_rank(self.hypo_id, doc_term_freq, events, top_n, thre) 
        for ranked_anchor in self.hypo_anchor_id: 
            self.hypo_anchor.append(events[str(ranked_anchor)].text)
            
        return ranked_doc_scores
           
    def graph_matching(self, events_dict, events_dict_tot=None, max_evi=10, num_hop=3, augment=False):
        
        def adjacency_matrix_matching(evidence_ls, events_dict, max_evi):
            
            adjacency_matrix = np.zeros((max_evi, max_evi))

            for ith,  event_1 in enumerate(evidence_ls):
                for jth,  event_2 in enumerate(evidence_ls):
                    try:
                        if event_2 in events_dict[str(event_1)].effects:
                            #adjacency_matrix[ith + 1, jth + 1] = 1
                            adjacency_matrix[ith, jth] = 1
                    except:
                        pass
                    try:
                        if event_2 in events_dict[str(event_1)].causes:
                            #adjacency_matrix[jth + 1, ith + 1] = 1
                            adjacency_matrix[jth, ith] = 1
                    except:
                        pass

            return adjacency_matrix                        

        def trunquate(ls, max_evi):
            '''
            if the length of evidence_id is shorter than max_evi, it will be paddedd in the data pretraining process before feed into NN.
            '''
            L = len(ls)
            diff = max_evi - L
            
            while diff < 0:
                pop_ith = random.randint(1, len(ls)-2)
                ls.pop(pop_ith)
                diff += 1

        def bfs(anchor, events_dict, direction='deductive', depth=3):
            neighborhood = []
            if depth == 1:
                if direction == 'deductive':
                    try:
                        return list(events_dict[str(anchor)].effects)
                    except:
                        return []
                  
            else:
                if direction == 'deductive':
                    try:
                        neighborhood = list(events_dict[str(anchor)].effects)
                    except:
                        return []
                else:
                    try:
                        neighborhood = list(events_dict[str(anchor)].causes)
                    except:
                        return []
                #pdb.set_trace()
                if len(neighborhood) > 0:
                    unvisited_neighbors = copy.deepcopy(neighborhood)
                    while len(unvisited_neighbors) > 0:
                        neighbor = unvisited_neighbors.pop()
                        #print(neighbor)
                        neighborhood += bfs(neighbor, events_dict, direction=direction, depth=depth-1)
                else:
                    neighborhood += []
                        
            return neighborhood        
                  
        evi_hypo = set()
        evi_alt = []
        
        for anchor_hypo in self.hypo_anchor_id:
            if self.ask_for == 1:
                # 1 for effect
                evi_tmp = bfs(anchor_hypo, events_dict, direction='deductive', depth=num_hop)
            else:
                evi_tmp = bfs(anchor_hypo, events_dict, direction='abductive', depth=num_hop)
             
            evi_hypo = evi_hypo.union(evi_tmp)
            #pdb.set_trace()
        
        for alt_anchor_id in self.alt_anchor_id:
            evi_alt.append(set())   
            for anchor_alt in alt_anchor_id:
                if self.ask_for == 1:
                    evi_tmp = bfs(anchor_alt, events_dict, direction='abductive', depth=num_hop)
                else:
                    evi_tmp = bfs(anchor_alt, events_dict, direction='abductive', depth=num_hop)
                
                evi_alt[-1] = evi_alt[-1].union(evi_tmp)
        
        for i in range(len(self.alt_id)):

            if not augment:
                evidence_id = list(evi_alt[i].intersection(evi_hypo))
            else:
                evidence_id = self.alt_anchor_id[i] + list(evi_alt[i].intersection(evi_hypo)) + self.hypo_anchor_id 
                    
            trunquate(evidence_id, max_evi)
            evidence_events = []
            for i in evidence_id:
                try:
                    evidence_events.append(events_dict[str(i)].text)
                except:
                    evidence_events.append(events_dict_tot[str(i)].text)
                
            self.evidence_id.append(evidence_id)
            self.evidence.append(evidence_events)        
        
        for evidence_id in self.evidence_id:
            adjacency_matrix_tmp = adjacency_matrix_matching(evidence_id, events_dict, max_evi)
            self.adjacency_matrix.append(adjacency_matrix_tmp)
       
        
class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 answer

    ):
        self.example_id = example_id
        try:
            self.choices_features = [
                {
                    'tokens': tokens,
                    'input_ids': input_ids,
                    'input_mask': input_mask,
                    'sentence_ind': sentence_ind,
                    'end_inds': end_inds
                }
                for tokens, input_ids, input_mask, sentence_ind, end_inds in choices_features
            ]   
        except:
            self.choices_features = [
                {
                    'tokens': tokens,
                    'input_ids': input_ids,
                    'input_mask': input_mask,
                    'sentence_ind': sentence_ind,
                }
                for tokens, input_ids, input_mask, sentence_ind in choices_features
            ]   

        self.answer = answer
