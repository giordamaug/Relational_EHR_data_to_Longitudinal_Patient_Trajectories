import pandas as pd
import numpy as np
from .models import FlexibleLSTMModel, LSTMDataset, lstm_collate_fn
from .models import GRUModel
from .models import DipoleDataset, DipoleModel, dipole_collate, Med2VecDataset, Med2VecModel, med2vec_collate
from collections import Counter
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder
from tqdm.notebook import tqdm
from IPython.display import clear_output

plotsize = (4,3)

def FlexLSTMclassifier(sequences,
                 labels, 
                 word_to_idx=None, 
                 train_idx=None, 
                 valid_idx=None,
                 num_epochs=10,
                 batch_size=32,
                 embed_size=64,
                 hidden_size=128,
                 threshold=0.3,
                 enable_plot=False,
                 frame_tqdm=None,
                 frame_plot=None,
                 pooling=False,
                 use_padding=False,
                 bidirectional=False,
                 use_attention=False,
                 name="FlexLSTM"
                 ):

    y_train = {id: val for (id,val) in labels.items() if id in train_idx} 
    y_valid = {id: val for (id,val) in labels.items() if id in valid_idx}
    train_sentences = {id: [word_to_idx[word] for word, _ in sequences[id]] for id in train_idx}
    train_dataset = LSTMDataset(train_sentences, labels_dict=y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=lstm_collate_fn)
    test_sentences = {id: [word_to_idx[word] for word, _ in sequences[id]] for id in valid_idx}
    test_dataset = LSTMDataset(test_sentences, labels_dict=y_valid)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lstm_collate_fn)

    model = FlexibleLSTMModel(len(word_to_idx), embed_dim=embed_size, 
                              hidden_dim=hidden_size, 
                              pooling=pooling,
                              use_padding=use_padding, 
                              bidirectional=bidirectional,
                              use_attention=use_attention, name=name)
    model.train_model(train_loader, val_loader=None, num_epochs=num_epochs, enable_plot=enable_plot, 
                      frame_tqdm=frame_tqdm, frame_plot=frame_plot, plotsize=plotsize)
    y_pred, y_prob = model.predict(test_loader, threshold=threshold)
    return y_pred, y_prob

def Med2Vecclassifier(sequences, word_to_idx, train_idx, valid_idx,
                          labels, embed_size=64, hidden_size=64, lambda_visit=0.1,
                          batch_size=32, num_epochs=10, threshold=0.3, 
                          enable_plot=False, frame_tqdm=None, frame_plot=None):


    train_sequences = {pid: sequences[pid] for pid in train_idx}
    valid_sequences = {pid: sequences[pid] for pid in valid_idx}

    train_ds = Med2VecDataset(train_sequences, labels, word_to_idx)
    valid_ds = Med2VecDataset(valid_sequences, labels, word_to_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False,collate_fn=med2vec_collate)
    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, shuffle=False,collate_fn=med2vec_collate)
    model = Med2VecModel(vocab_size=len(word_to_idx),embed_dim=embed_size, visit_dim=64)
    model.train_model(train_loader, val_loader=None, num_epochs=num_epochs, lambda_visit=lambda_visit, enable_plot=enable_plot, 
                      frame_tqdm=frame_tqdm, frame_plot=frame_plot, plotsize=plotsize)
    y_pred, y_prob = model.predict(valid_loader, threshold=threshold)
    return y_pred, y_prob

def DipoleClassifier(sequences,
                word_to_idx=None,
                train_idx=None,
                valid_idx=None,
                batch_size=32,
                labels=None,
                embed_size=64,
                hidden_size=128,
                num_epochs=10,
                threshold=0.3,
                enable_plot=False,
                frame_tqdm=None,
                frame_plot=None):

    train_visists = {id:v for id,v in sequences.items() if id in train_idx}
    train_dataset = DipoleDataset(train_visists, labels=labels, code2id=word_to_idx)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=dipole_collate)
    test_visists = {id:v for id,v in sequences.items() if id in valid_idx}
    test_dataset = DipoleDataset(test_visists, labels=None, code2id=word_to_idx)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=dipole_collate)
    
    model = DipoleModel(vocab_size=len(word_to_idx),embed_size=embed_size, hidden_size=hidden_size)
    model.train_model(train_loader, val_loader=None, num_epochs=num_epochs, enable_plot=enable_plot, 
                      frame_tqdm=frame_tqdm, frame_plot=frame_plot, plotsize=plotsize)
    y_pred, y_prob = model.predict(test_loader, threshold=threshold)
    return y_pred, y_prob

def GRUclassifier(sequences,
                word_to_idx=None,
                train_idx=None,
                valid_idx=None,
                batch_size=32,
                labels=None,
                embed_size=64,
                hidden_size=128,
                num_epochs=10,
                threshold=0.3,
                enable_plot=False,
                frame_tqdm=None,
                frame_plot=None):
    
    y_train = {id: val for (id,val) in labels.items() if id in train_idx} 
    y_valid = {id: val for (id,val) in labels.items() if id in valid_idx}
    train_sentences = {id: [word_to_idx[word] for word, _ in sequences[id]] for id in train_idx}
    train_dataset = LSTMDataset(train_sentences, labels_dict=y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=lstm_collate_fn)
    test_sentences = {id: [word_to_idx[word] for word, _ in sequences[id]] for id in valid_idx}
    test_dataset = LSTMDataset(test_sentences, labels_dict=y_valid)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lstm_collate_fn)

    model = GRUModel(len(word_to_idx), embed_dim=embed_size, gru_hidden_dim=hidden_size, pooling="mean")
    model.train_model(train_loader, val_loader=None, num_epochs=num_epochs, enable_plot=enable_plot, 
                      frame_tqdm=frame_tqdm, frame_plot=frame_plot, plotsize=plotsize)
    y_pred, y_prob = model.predict(test_loader, threshold=threshold)
    return y_pred, y_prob


