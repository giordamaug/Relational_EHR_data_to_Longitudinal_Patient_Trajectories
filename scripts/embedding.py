import pandas as pd
import numpy as np
from .models import FlexibleLSTMModel, LSTMDataset, lstm_collate_fn
from .models import BEHRTDataset, BEHRTModel
from .models import GRUModel, GRUDDataset, GRUDModel, grud_collate_fn
from .models import DipoleDataset, DipoleModel, dipole_collate
from .models import DOME, co_occurrence_infectious_window,compute_directional_ppmi, riskmatrix_loop_fb_dome
from .models import Med2VecDataset, Med2VecModel, med2vec_collate
from .models import CEHRBERTModel,CEHRBERTDataset, cehrbert_collate_fn
from collections import Counter
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder
from tqdm.notebook import tqdm
from IPython.display import clear_output
from .utils import extract_event_type_counters

plotsize = (4,3)

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pandas as pd


def FlexLSTMembeddeWithValidationr(
    sequences,
    labels,
    word_to_idx=None,
    train_idx=None,
    valid_idx=None,
    num_epochs=50,
    batch_size=32,
    embed_size=64,
    hidden_size=128,
    enable_plot=False,
    frame_tqdm=None,
    frame_plot=None,
    pooling=False,
    use_padding=False,
    bidirectional=False,
    use_attention=False,
    name="FlexLSTM",
    lstm_val_size=0.2,
    random_state=42,
):
    """
    train_idx: pazienti usati per addestrare l'embedder
    valid_idx: pazienti esterni, usati solo per estrarre embedding dopo il training
    """

    # Split interno SOLO dentro il train
    y_for_split = [labels[i] for i in train_idx]

    lstm_train_idx, lstm_val_idx = train_test_split(
        list(train_idx),
        test_size=lstm_val_size,
        stratify=y_for_split,
        random_state=random_state,
    )

    # Labels
    y_lstm_train = {i: labels[i] for i in lstm_train_idx}
    y_lstm_val = {i: labels[i] for i in lstm_val_idx}
    y_valid = {i: labels[i] for i in valid_idx}

    # Sequenze indicizzate
    lstm_train_sentences = {
        i: [word_to_idx[word] for word, _ in sequences[i] if word in word_to_idx]
        for i in lstm_train_idx
    }

    lstm_val_sentences = {
        i: [word_to_idx[word] for word, _ in sequences[i] if word in word_to_idx]
        for i in lstm_val_idx
    }

    valid_sentences = {
        i: [word_to_idx[word] for word, _ in sequences[i] if word in word_to_idx]
        for i in valid_idx
    }

    # Dataset
    lstm_train_dataset = LSTMDataset(
        lstm_train_sentences,
        labels_dict=y_lstm_train
    )

    lstm_val_dataset = LSTMDataset(
        lstm_val_sentences,
        labels_dict=y_lstm_val
    )

    valid_dataset = LSTMDataset(
        valid_sentences,
        labels_dict=y_valid
    )

    # DataLoader
    lstm_train_loader = DataLoader(
        lstm_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lstm_collate_fn
    )

    lstm_val_loader = DataLoader(
        lstm_val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lstm_collate_fn
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lstm_collate_fn
    )

    # Modello
    model = FlexibleLSTMModel(
        len(word_to_idx),
        embed_dim=embed_size,
        hidden_dim=hidden_size,
        pooling=pooling,
        use_padding=use_padding,
        bidirectional=bidirectional,
        use_attention=use_attention,
        name=name
    )

    # Training con validation interna
    model.train_model(
        lstm_train_loader,
        val_loader=lstm_val_loader,
        num_epochs=num_epochs,
        enable_plot=enable_plot,
        frame_tqdm=frame_tqdm,
        frame_plot=frame_plot
    )
    # Ora estrai embedding su TUTTO il train esterno
    full_train_labels = {i: labels[i] for i in train_idx}

    full_train_sentences = {
        i: [word_to_idx[word] for word, _ in sequences[i] if word in word_to_idx]
        for i in train_idx
    }

    full_train_dataset = LSTMDataset(
        full_train_sentences,
        labels_dict=full_train_labels
    )

    full_train_loader = DataLoader(
        full_train_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lstm_collate_fn
    )

    train_patient_embeddings, train_ids = model.get_embeddings(full_train_loader)
    valid_patient_embeddings, valid_ids = model.get_embeddings(valid_loader)

    # DataFrame
    train_df = pd.DataFrame(train_patient_embeddings, index=train_ids)
    valid_df = pd.DataFrame(valid_patient_embeddings, index=valid_ids)

    train_df.columns = [f"{name}_{i}" for i in range(train_df.shape[1])]
    valid_df.columns = [f"{name}_{i}" for i in range(valid_df.shape[1])]

    return train_df, valid_df
    
def FlexLSTMembedder(sequences,
                 labels, 
                 word_to_idx=None, 
                 train_idx=None, 
                 valid_idx=None,
                 num_epochs=10,
                 batch_size=32,
                 embed_size=64,
                 hidden_size=128,
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

    train_patient_embeddings, tids = model.get_embeddings(train_loader)
    test_patient_embeddings, tsids = model.get_embeddings(test_loader)
    
    train_df = pd.DataFrame(train_patient_embeddings, index=train_idx)
    test_df = pd.DataFrame(test_patient_embeddings, index=valid_idx)
    train_df.columns = [f"{name}_{i}" for i in range(train_df.shape[1])]
    test_df.columns = [f"{name}_{i}" for i in range(test_df.shape[1])]
    return train_df, test_df

def RETAINembedder(sequences, 
                    labels,
                    word_to_idx=None, 
                    train_idx=None, 
                    valid_idx=None,
                    num_epochs=10,
                    batch_size=32,
                    embed_size=64,
                    hidden_size=128,
                    enable_plot=False,
                    frame_tqdm=None,
                    frame_plot=None,
                 ):
    train_sequences = [[[word_to_idx[word] for word in words] for words,_ in sequences[id]] for id in train_idx]
    test_sequences = [[[word_to_idx[word] for word in words] for words,_ in sequences[id]] for id in valid_idx]
    y_train = np.array([val for (id,val) in labels.items() if id in train_idx], dtype=np.float32)
    y_valid = np.array([val for (id,val) in labels.items() if id in valid_idx], dtype=np.float32)
    
    #max_code = max(map(lambda p: max(map(lambda v: max(v), p)), train_sequences + test_sequences))
    num_features = len(word_to_idx) #max_code + 1
    # Inizializza e allena modello LSTM unidirezionale
    model = RETAINModel(len(word_to_idx), dim_emb=embed_size, dim_alpha=hidden_size, dim_beta=hidden_size)
    test_dataset = RETAINDataset(test_sequences, y_valid, num_features)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, collate_fn=visit_collate_fn)
    train_dataset = RETAINDataset(train_sequences, y_train, num_features)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, collate_fn=visit_collate_fn)
    model.train_model(train_dataloader, val_loader=None, num_epochs=num_epochs, enable_plot=enable_plot, 
                      frame_tqdm=frame_tqdm, frame_plot=frame_plot, plotsize=plotsize)
    
    train_patient_embeddings, tids = model.get_embeddings(train_dataloader)
    test_patient_embeddings, tsids = model.get_embeddings(test_dataloader)
    train_df = pd.DataFrame(train_patient_embeddings, index=train_idx)
    test_df = pd.DataFrame(test_patient_embeddings, index=valid_idx)
    train_df.columns = [f"retain_{i}" for i in range(train_df.shape[1])]
    test_df.columns = [f"retain_{i}" for i in range(test_df.shape[1])]
    return train_df, test_df

def CEHRBERTembedder(
    sequences,
    labels,
    tmapper,
    word_to_idx=None,
    train_idx=None,
    valid_idx=None,
    event_type_dict=None,
    with_validation=True,
    num_epochs=10,
    batch_size=32,
    embed_size=64,
    hidden_size=128,
    num_heads=4,
    num_layers=2,
    max_len=512,
    pooling="attention",
    use_token_type=True,
    loss_type="focal",
    alpha=0.85,
    gamma=2.0,
    patience=5,
    lr=1e-4,
    weight_decay=1e-5,
    random_state=42,
    encoder_val_size=0.2,
    enable_plot=False,
    frame_tqdm=None,
    frame_plot=None,
    plotsize=(6, 4),
    name="CEHRBERT"
):
    train_visists = {id:v for id,v in sequences.items() if id in train_idx}
    test_visists = {id:v for id,v in sequences.items() if id in valid_idx}
    full_train_event_type_dict = None #{id: [tmapper[x[2]] for x in events] for id,events in sequences.items() if id in train_idx}
    full_train_dataset = CEHRBERTDataset(train_visists, labels_dict=labels,event_type_dict=full_train_event_type_dict,max_len=max_len, word_to_idx=word_to_idx)
    full_train_dataloader = DataLoader(dataset=full_train_dataset, batch_size=batch_size, collate_fn=cehrbert_collate_fn)

    train_idx = list(train_idx)
    valid_idx = list(valid_idx)

    # split interno SOLO per addestrare/validare l'encoder
    y_train_split = [labels[i] for i in train_idx]
    encoder_train_idx, encoder_val_idx = train_test_split(
        train_idx,
        test_size=encoder_val_size,
        stratify=y_train_split,
        random_state=random_state
    )

    model = CEHRBERTModel(vocab_size=len(word_to_idx),embed_dim=embed_size, hidden_dim=hidden_size,
                          num_heads=num_heads,num_layers=num_layers,max_len=max_len,pooling=pooling,
                          use_token_type=use_token_type,name=name)
    if with_validation:
        train_visits = {id:v for id,v in sequences.items() if id in encoder_train_idx}
        val_visits = {id:v for id,v in sequences.items() if id in encoder_val_idx}
        train_dataset = CEHRBERTDataset(train_visits, labels_dict=labels,event_type_dict=event_type_dict,max_len=max_len, word_to_idx=word_to_idx)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, collate_fn=cehrbert_collate_fn)
        val_dataset = CEHRBERTDataset(val_visits, labels_dict=labels,event_type_dict=event_type_dict,max_len=max_len, word_to_idx=word_to_idx)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, collate_fn=cehrbert_collate_fn) 
    else:
        #train_dataset = CEHRBERTDataset(train_visists, labels_dict=labels,event_type_dict=event_type_dict,max_len=max_len, word_to_idx=word_to_idx)
        #train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, collate_fn=cehrbert_collate_fn)
        train_dataloader = full_train_dataloader
        val_dataloader = None

    test_event_type_dict = None #{id: [tmapper[x[2]] for x in events] for id,events in sequences.items() if id in valid_idx}
    test_dataset = CEHRBERTDataset(test_visists, event_type_dict=test_event_type_dict,max_len=max_len, word_to_idx=word_to_idx)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, collate_fn=cehrbert_collate_fn)

    model.train_model(train_dataloader,val_loader=val_dataloader,num_epochs=num_epochs, lr=lr,
                    loss_type=loss_type,alpha=alpha,gamma=gamma,weight_decay=weight_decay,
                    enable_plot=enable_plot,frame_tqdm=frame_tqdm,frame_plot=frame_plot, plotsize=plotsize)

    train_embeddings, train_ids = model.get_embeddings(full_train_dataloader)
    valid_embeddings, valid_ids = model.get_embeddings(test_dataloader)

    train_df = pd.DataFrame(train_embeddings, index=train_ids)
    valid_df = pd.DataFrame(valid_embeddings, index=valid_ids)

    train_df.columns = [f"{name}_{i}" for i in range(train_df.shape[1])]
    valid_df.columns = [f"{name}_{i}" for i in range(valid_df.shape[1])]

    return train_df, valid_df#, model, history

def BEHRTembedder(sequences, 
                  labels,
                  word_to_idx=None, 
                  train_idx=None, 
                  valid_idx=None,
                  num_epochs=10,
                  batch_size=32,
                  embed_size=64,
                  hidden_size=128,
                  enable_plot=False,
                  frame_tqdm=None,
                  frame_plot=None,
                 ):
    train_visists = {id:v for id,v in sequences.items() if id in train_idx}
    test_visists = {id:v for id,v in sequences.items() if id in valid_idx}

    train_dataset = BEHRTDataset(train_visists, labels_dict=labels, code2id=word_to_idx)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size)
    test_dataset = BEHRTDataset(test_visists, code2id=word_to_idx)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size)

    model = BEHRTModel(vocab_size=len(word_to_idx),embed_dim=embed_size,num_layers=2, num_heads=4, num_labels=2)

    model.train_model(train_dataloader, val_loader=None, num_epochs=num_epochs, lr=1e-4, enable_plot=enable_plot, 
                      frame_tqdm=frame_tqdm, frame_plot=frame_plot, plotsize=plotsize)
    train_patient_embeddings = model.get_embeddings(train_dataloader)
    test_patient_embeddings = model.get_embeddings(test_dataloader)
    train_df = pd.DataFrame(train_patient_embeddings, index=train_idx)
    test_df = pd.DataFrame(test_patient_embeddings, index=valid_idx)
    train_df.columns = [f"behrt_{i}" for i in range(train_df.shape[1])]
    test_df.columns = [f"behrt_{i}" for i in range(test_df.shape[1])]
    return train_df, test_df

def DOMEEmbedder(sequences,
                 targets,
                 df,
                 train_idx=None,
                 valid_idx=None,
                 enable_plot=False,
                 frame_tqdm=None,
                 frame_plot=None,
                ):
    all_idx = np.concatenate((train_idx, valid_idx), axis=0)
    events_fold_train = {id: sequences[int(id)] for id in train_idx if int(id) in sequences}
    with frame_tqdm:
        frame_tqdm.clear_output(wait=True)
        cooc_prior, vocabular = co_occurrence_infectious_window(events_fold_train, targets, df, months_window=5, direction='prior') #, exclude_from_rows=set(targets))
        cooc_post, _ = co_occurrence_infectious_window(events_fold_train, targets, df, months_window=5, direction='posterior') #, exclude_from_rows=set(targets))
    with frame_plot:
        frame_plot.clear_output(wait=True)
        print(f"✅ Vocabulary with {len(vocabular)} words (first 10 words):", vocabular[:10])
        print("📊 Co-occurrence matrix shape:", cooc_prior.shape, cooc_post.shape)
    
    embed_attributes = [w for w in sorted(vocabular) if w not in targets]

    P_plus = compute_directional_ppmi(cooc_post.values)
    P_minus = compute_directional_ppmi(cooc_prior.values)
    W, C_plus, C_minus = DOME().fit(P_plus, P_minus)

    vocab_lower = [w.lower() for w in cooc_prior.columns]
    W_df = pd.DataFrame(W.T, columns=vocab_lower)
    C_df = pd.DataFrame(C_minus.T, columns=vocab_lower)
    event_counts_all = {}
    for id, timelines in sequences.items():
        eventi = [ev for ev, _ in timelines if isinstance(ev, str)]
        filtrati = [e.lower().strip() for e in eventi if e.lower().strip() != "<ph>"]
        event_counts_all[int(id)] = dict(Counter(filtrati))

    valid_tokens = set(W_df.columns).intersection(C_df.columns).intersection(embed_attributes)
    with frame_tqdm:
        Xrisk = riskmatrix_loop_fb_dome(
            events=event_counts_all,
            attributes=[a for a in embed_attributes if a in valid_tokens],
            targets=[t for t in targets if t in C_df.columns],
            emb1=W_df,
            emb2=C_df
        )
    Xrisk.index = Xrisk.index.astype(int)
    Xembed_dome = Xrisk.loc[Xrisk.index.intersection([int(id) for id in all_idx])]
    Xembed_dome.columns = [f"{col}_embed" for col in Xembed_dome.columns]
    with frame_plot:
        print(f"✅ Embedding size {len(Xembed_dome.columns)}")

    train_df = pd.DataFrame(Xembed_dome.loc[train_idx], index=train_idx)
    test_df = pd.DataFrame(Xembed_dome.loc[valid_idx], index=valid_idx)
    train_df.columns = [f"dome_{i}" for i in range(train_df.shape[1])]
    test_df.columns = [f"dome_{i}" for i in range(test_df.shape[1])]
    return train_df, test_df

def StaticEmbedder(df,
                   train_idx=None, 
                   valid_idx=None,
                   enable_plot=False,
                    frame_tqdm=None,
                    frame_plot=None,
                   ):
    return df.loc[train_idx], df.loc[valid_idx]

import re
def rename_cols(df):
    # Change columns names ([LightGBM] Do not support special JSON characters in feature name.)
    new_names = {col: re.sub(r'[^A-Za-z0-9_]+', '', col) for col in df.columns}
    new_n_list = list(new_names.values())
    # [LightGBM] Feature appears more than one time.
    new_names = {col: f'{new_col}_{i}' if new_col in new_n_list[:i] else new_col for i, (col, new_col) in enumerate(new_names.items())}
    return df.rename(columns=new_names)
    
def BINARYEmbedder(sequences,
                   targets,
                   vocab=None,
                   train_idx=None,
                   valid_idx=None,
                   enable_plot=False,
                   frame_tqdm=None,
                   frame_plot=None,
                    ):
    all_idx = np.concatenate((train_idx, valid_idx), axis=0)
    zdata = np.zeros(shape=(len(all_idx),len(vocab)))
    df_bin = pd.DataFrame(zdata, columns=list(vocab), index=all_idx)
    with frame_tqdm:
        frame_tqdm.clear_output(wait=True)
        for id,events in tqdm(sequences.items(), total=len(sequences), desc="[BINARY] contruct:"):
            for event,date in events:
                df_bin.loc[id][event] = 1
    df_bin = rename_cols(df_bin)
    train_df = df_bin.loc[train_idx]
    test_df = df_bin.loc[valid_idx]
    return train_df, test_df

def FREQEmbedder(sequences,
                   targets,
                   vocab=None,
                   train_idx=None,
                   valid_idx=None,
                   enable_plot=False,
                   frame_tqdm=None,
                   frame_plot=None,
                    ):
    all_idx = np.concatenate((train_idx, valid_idx), axis=0)
    zdata = np.zeros(shape=(len(all_idx),len(vocab)))
    df_bin = pd.DataFrame(zdata, columns=list(vocab), index=all_idx)
    with frame_tqdm:
        frame_tqdm.clear_output(wait=True)
        for id,events in tqdm(sequences.items(), total=len(sequences), desc="[BINARY] contruct:"):
            for event,date in events:
                df_bin.loc[id][event] += 1
    df_bin = rename_cols(df_bin)
    train_df = df_bin.loc[train_idx]
    test_df = df_bin.loc[valid_idx]
    return train_df, test_df

def COUNTEmbedder(sequences,
                   train_idx=None,
                   valid_idx=None,
                   enable_plot=False,
                   frame_tqdm=None,
                   frame_plot=None,
                    ):
    
    df_cnt = extract_event_type_counters(sequences)
    train_df = df_cnt.loc[train_idx]
    test_df = df_cnt.loc[valid_idx]
    if enable_plot:
        print(f"Counting event types: {' '.join(list(df_cnt.columns))}")
    return train_df, test_df

def GRUEmbedder(sequences,
                word_to_idx=None,
                train_idx=None,
                valid_idx=None,
                batch_size=32,
                labels=None,
                embed_size=64,
                hidden_size=128,
                num_epochs=10,
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

    #embeddings, ids = model.get_embeddings(train_loader)
    train_patient_embeddings, tids = model.get_embeddings(train_loader)
    test_patient_embeddings, tsids = model.get_embeddings(test_loader)
    
    train_df = pd.DataFrame(train_patient_embeddings, index=train_idx)
    test_df = pd.DataFrame(test_patient_embeddings, index=valid_idx)
    train_df.columns = [f"gru_{i}" for i in range(train_df.shape[1])]
    test_df.columns = [f"gru_{i}" for i in range(test_df.shape[1])]
    return train_df, test_df

def DipoleEmbedder(sequences,
                word_to_idx=None,
                train_idx=None,
                valid_idx=None,
                batch_size=32,
                labels=None,
                embed_size=64,
                hidden_size=128,
                num_epochs=10,
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
    
    train_patient_embeddings, tids = model.get_embeddings(train_loader)
    test_patient_embeddings, tsids = model.get_embeddings(test_loader)
    train_df = pd.DataFrame(train_patient_embeddings, index=train_idx)
    test_df = pd.DataFrame(test_patient_embeddings, index=valid_idx)
    train_df.columns = [f"dipole_{i}" for i in range(train_df.shape[1])]
    test_df.columns = [f"dipole_{i}" for i in range(test_df.shape[1])]
    return train_df, test_df


def GRUEDembedder(sequences,
                word_to_idx=None,
                train_idx=None,
                valid_idx=None,
                batch_size=32,
                labels=None,
                embed_size=64,
                hidden_size=128,
                num_epochs=10,
                enable_plot=False,
                frame_tqdm=None,
                frame_plot=None):
    train_visists = {id:v for id,v in sequences.items() if id in train_idx}
    train_dataset = GRUDDataset(train_visists, labels_dict=labels, code2id=word_to_idx, max_seq_len=50)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=grud_collate_fn)
    test_visists = {id:v for id,v in sequences.items() if id in valid_idx}
    test_dataset = GRUDDataset(test_visists, code2id=word_to_idx)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=grud_collate_fn)

    model = GRUDModel(input_size=len(word_to_idx), 
                        hidden_size=hidden_size, output_size=2, 
                        x_mean=train_dataset.feature_means)
    model.train_model(train_loader, val_loader=None, num_epochs=num_epochs, enable_plot=enable_plot, 
                      frame_tqdm=frame_tqdm, frame_plot=frame_plot, plotsize=plotsize)

    #embeddings, ids = model.get_embeddings(train_loader)
    train_patient_embeddings, tids = model.get_embeddings(train_loader)
    test_patient_embeddings, tsids = model.get_embeddings(test_dataloader)
    
    train_df = pd.DataFrame(train_patient_embeddings, index=train_idx)
    test_df = pd.DataFrame(test_patient_embeddings, index=valid_idx)
    train_df.columns = [f"gru-d_{i}" for i in range(train_df.shape[1])]
    test_df.columns = [f"gru-d_{i}" for i in range(test_df.shape[1])]
    return train_df, test_df


def TimeAwareLSTMEmbedder(sequences, word_to_idx, train_idx, valid_idx,
                          labels, embed_size=64, hidden_size=64,
                          batch_size=32, num_epochs=10, enable_plot=False,
                          frame_tqdm=None, frame_plot=None):


    train_sequences = {pid: sequences[pid] for pid in train_idx}
    valid_sequences = {pid: sequences[pid] for pid in valid_idx}

    train_ds = TimeAwareLSTMDataset(train_sequences, labels, word_to_idx)
    valid_ds = TimeAwareLSTMDataset(valid_sequences, labels, word_to_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False,collate_fn=timeaware_collate_fn)
    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, shuffle=False,collate_fn=timeaware_collate_fn)
    model = TimeAwareLSTMModel(vocab_size=len(word_to_idx),embed_size=embed_size,hidden_size=hidden_size)
    model.train_model(train_loader, val_loader=None, num_epochs=num_epochs, enable_plot=enable_plot, 
                      frame_tqdm=frame_tqdm, frame_plot=frame_plot, plotsize=plotsize)
    train_emb, tids = model.get_embeddings(train_loader)
    valid_emb, vids = model.get_embeddings(valid_loader)

    df_train = pd.DataFrame(train_emb, index=train_idx)
    df_valid = pd.DataFrame(valid_emb, index=valid_idx)
    df_train.columns = [f"tLSTM_{i}" for i in range(df_train.shape[1])]
    df_valid.columns = [f"tLSTM_{i}" for i in range(df_valid.shape[1])]

    return df_train, df_valid

def Med2VecEmbedder(sequences, word_to_idx, train_idx, valid_idx,
                          labels, embed_size=64, hidden_size=64, lambda_visit=0.1,
                          batch_size=32, num_epochs=10, enable_plot=False,
                          frame_tqdm=None, frame_plot=None):


    train_sequences = {pid: sequences[pid] for pid in train_idx}
    valid_sequences = {pid: sequences[pid] for pid in valid_idx}

    train_ds = Med2VecDataset(train_sequences, labels, word_to_idx)
    valid_ds = Med2VecDataset(valid_sequences, labels, word_to_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False,collate_fn=med2vec_collate)
    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, shuffle=False,collate_fn=med2vec_collate)
    model = Med2VecModel(vocab_size=len(word_to_idx),embed_dim=embed_size, visit_dim=64)
    model.train_model(train_loader, val_loader=None, num_epochs=num_epochs, lambda_visit=lambda_visit, enable_plot=enable_plot, 
                      frame_tqdm=frame_tqdm, frame_plot=frame_plot, plotsize=plotsize)
    train_emb, tids = model.get_embeddings(train_loader)
    valid_emb, vids = model.get_embeddings(valid_loader)

    df_train = pd.DataFrame(train_emb, index=train_idx)
    df_valid = pd.DataFrame(valid_emb, index=valid_idx)
    df_train.columns = [f"M2V_{i}" for i in range(df_train.shape[1])]
    df_valid.columns = [f"M2V_{i}" for i in range(df_valid.shape[1])]

    return df_train, df_valid
