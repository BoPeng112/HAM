import pickle
import pdb
from scipy.stats import pearsonr
import numpy as np

with open('CDs_item_sequences.pkl', 'rb') as f:
    item_lists = pickle.load(f)

with open('att_dict.pkl', 'rb') as f:
    att = pickle.load(f)

with open('time_dict.pkl', 'rb') as f:
    time = pickle.load(f)

freq_dict = {}
#index shift by 1 in the model
for user_list in item_lists:
    for item in user_list:

        freq_dict[item + 1] = freq_dict.get(item + 1, 0) + 1

normalized_weight = []
selected_freq = []

for i in range(1, len(freq_dict) + 1):

    if i in att:
        normalized_weight.append(att[i] / time[i])
        selected_freq.append(freq_dict[i])

normalized_weight = np.asarray(normalized_weight)
selected_freq = np.asarray(selected_freq)

corr, _ = pearsonr(normalized_weight, selected_freq)

pdb.set_trace()

print(corr)
