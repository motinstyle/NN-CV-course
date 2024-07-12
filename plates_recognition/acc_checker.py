import pandas as pd
import numpy as np
import os
import sys

def check_acc(submission):
    # cur_df = pd.read_csv(sys.argv[1])
    cur_df = pd.read_csv(submission)
    sample_df = pd.read_csv('my_submission.csv')

    cur_labels = cur_df.get('label')
    sample_labels = sample_df.get('label')

    np_cur = np.array(cur_labels)
    np_sample = np.array(sample_labels)

    sum = np.sum((np_cur == np_sample).astype(float))

    accuracy = sum/len(os.listdir("./test/unknown"))
    print(accuracy)

# cur_df = pd.read_csv(sys.argv[1])
# sample_df = pd.read_csv('my_submission.csv')
# cur_labels = cur_df.get('label')
# sample_labels = sample_df.get('label')
# np_cur = np.array(cur_labels)
# np_sample = np.array(sample_labels)
# sum = np.sum((np_cur == np_sample).astype(float))
# accuracy = sum/len(os.listdir("./test/unknown"))
# print(accuracy)