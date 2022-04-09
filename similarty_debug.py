import pandas as pd

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
for index, row in merged_ques_facts_df.iterrows():
    if loopcount == 2:
        break
    loopcount+=1
    topK_similar_fact_ids, topK_similar_fact_scores = fact_similarity(row)
    top_100_similar_fact_ids.append(topK_similar_fact_ids)
    top_100_similar_fact_scores.append(topK_similar_fact_scores)