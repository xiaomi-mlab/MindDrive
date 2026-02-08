import pickle

def load_pkl_data(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data