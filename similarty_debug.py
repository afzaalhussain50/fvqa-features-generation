import os

import numpy as np
import pandas as pd

embeddings_dict = {}
path_glove300d = "F:\\NUST\\thesis_local\\Practice\\straight-to-the-fact\\glove.6B"
with open(os.path.join(path_glove300d, "glove.6B.300d.txt"), 'r', encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector

from string import punctuation
def question_glove_embedding(question):
    question = ''.join([q for q in question if q not in punctuation])
    question = question.split()
    question_words_emb = []
    for w in question:
        try:
            question_words_emb.append(embeddings_dict[w.lower()])
        except KeyError:
            question_words_emb.append(np.zeros(300))
    data = np.array(question_words_emb)
    question_glove_emb = np.average(data, axis=0)
    return question_glove_emb

def visual_concept_glove_embedding(vc):
    vc = ' '.join([word for word in vc if word not in punctuation])
    vc = vc.replace('_', ' ')
    vc = vc.replace('/', ' ')
    vc_splits = vc.split()
    vc_words_emb = []
    for w in vc_splits:
        try:
            vc_words_emb.append(embeddings_dict[w.lower()])
        except KeyError:
            vc_words_emb.append(np.zeros(300))
    data = np.array(vc_words_emb)
    visual_concepts_glove_emb = np.average(data, axis=0)
    return visual_concepts_glove_emb

def fact_glove_embeddings(fact):
    e1_labels = fact['e1_label']
    e1_labels = ''.join([label for label in e1_labels if label not in punctuation])
    fact_words = e1_labels.split()
    e2_labels = fact['e2_label']
    e2_labels = ''.join([label for label in e2_labels if label not in punctuation])
    fact_words.extend(e2_labels.split())
    fact_emb = []
    for w in fact_words:
        try:
            fact_emb.append(embeddings_dict[w.lower()])
        except KeyError:
            fact_emb.append(np.zeros(300))
    fact_emb = np.array(fact_emb)
    fact_emb = np.average(fact_emb, axis=0)
    return fact_emb


from scipy import spatial
def calculate_cosine_distance(a, b):
    cosine_distance = float(spatial.distance.cosine(a, b))
    return cosine_distance
def calculate_cosine_similarity(a, b):
    cosine_similarity = 1 - calculate_cosine_distance(a, b)
    return cosine_similarity

merged_ques_facts_df = pd.read_pickle('merged_ques_facts_df.pkl')
all_facts_data_df = pd.read_pickle('all_facts_data_df.pkl')
def fact_similarity(merged_df_row):
    K= 100
    fact_similarity_scores = []
    for index, fact_row in all_facts_data_df.iterrows():

        fact_similarity_scores.append(calculate_cosine_similarity(merged_df_row['vc_ques_avg_emb'],
                                                                             fact_row['glove_emb']))
    all_facts_data_df['similarty_scores'] = fact_similarity_scores
    sorted_facts_df = all_facts_data_df.sort_values(by=['similarty_scores'], ascending=False)
    topKfacts = sorted_facts_df.iloc[0: K]
    print(merged_df_row)
    print(topKfacts)
    return topKfacts.index.tolist(), topKfacts['similarty_scores'].tolist()

top_100_similar_fact_ids = []
top_100_similar_fact_scores = []
loopcount = 1
# for index, row in merged_ques_facts_df.iterrows():
#     if loopcount == 2:
#         break
#     loopcount+=1
#     topK_similar_fact_ids, topK_similar_fact_scores = fact_similarity(row)
#     top_100_similar_fact_ids.append(topK_similar_fact_ids)
#     top_100_similar_fact_scores.append(topK_similar_fact_scores)

first_ques = merged_ques_facts_df.iloc[0]
most_similar_fact = all_facts_data_df.loc['conceptnet/e/bdb278197e7b379d787a9fb0fd24688a73a44da9']
supporting_similar_fact = all_facts_data_df.loc['"conceptnet/e/f768f157e4446dd594536f8ef02681515586ba2d"']
ques_emb = question_glove_embedding(first_ques['question'])
vc_emb = visual_concept_glove_embedding(first_ques['detected_visual_concepts'])
vc_ques_emb = np.average([ques_emb, vc_emb], axis=0)

most_similar_fact_emb = fact_glove_embeddings(most_similar_fact)
supporting_similar_fact = fact_glove_embeddings(supporting_similar_fact)

most_similar_fact_similarity = calculate_cosine_similarity(vc_ques_emb, most_similar_fact_emb)
supporting_similar_fact_similarity = calculate_cosine_similarity(vc_ques_emb, supporting_similar_fact)
a=10