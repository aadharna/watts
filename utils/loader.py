import argparse
import yaml
import os
import pickle

def load_from_yaml(fpath):

    args = yaml.load(open(fpath), Loader=yaml.FullLoader)
    return argparse.Namespace(**args)

def save_obj(obj, folder, name):

    path = os.path.join(folder, name) + '.pkl'
    if os.path.exists(path):
        os.remove(path)
    with open(path, 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(folder, name):

    path = os.path.join(folder, name)
    try:
        with open(path, 'rb') as f:
            f = pickle.load(f)
    except EOFError:
        f = {"resend":True}
    return f
