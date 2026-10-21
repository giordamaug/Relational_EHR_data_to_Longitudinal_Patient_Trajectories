import pandas as pd
import numpy as np
from .models import FlexibleLSTMModel, LSTMDataset, lstm_collate_fn
from .models import BEHRTDataset, BEHRTModel
from .models import GRUModel
from .models import DipoleDataset, DipoleModel, dipole_collate
from .models import DOME, co_occurrence_infectious_window,compute_directional_ppmi, riskmatrix_loop_fb_dome
from .models import Med2VecDataset, Med2VecModel, med2vec_collate
from .models import CEHRBERTModel,CEHRBERTDataset, cehrbert_collate_fn
from .models import TFIDFModel, BOWModel
from collections import Counter
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder
from tqdm.notebook import tqdm
from IPython.display import clear_output
from .utils import extract_event_type_counters, extract_event_counters
from .utils import count_total_parameters, count_trainable_parameters, MemoryProfiler, reset_gpu_memory, get_peak_gpu_memory_mb, get_process_memory_mb
import time

plotsize = (4,3)

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pandas as pd
from scripts.utils import fill_profile


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

    reset_gpu_memory()                          # Perf measure
    #cpu_mem_before = get_process_memory_mb()    # Perf measure
    t0 = time.perf_counter()                    # Perf measure
    model = FlexibleLSTMModel(len(word_to_idx), embed_dim=embed_size, 
                              hidden_dim=hidden_size, 
                              pooling=pooling,
                              use_padding=use_padding, 
                              bidirectional=bidirectional,
                              use_attention=use_attention, name=name)
    print(model)
    model.train_model(train_loader, val_loader=None, num_epochs=num_epochs, enable_plot=enable_plot, 
                      frame_tqdm=frame_tqdm, frame_plot=frame_plot, plotsize=plotsize)
    if torch.cuda.is_available(): torch.cuda.synchronize()     # Perf measure
    train_time = time.perf_counter() - t0                      # Perf measure
    if torch.cuda.is_available(): torch.cuda.synchronize()     # Perf measure
 
    t0 = time.perf_counter()     # Perf measure
    train_patient_embeddings, tids = model.get_embeddings(train_loader)
    test_patient_embeddings, tsids = model.get_embeddings(test_loader)
    
    if torch.cuda.is_available(): torch.cuda.synchronize()     # Perf measure
    inference_time = time.perf_counter() - t0                  # Perf measure
    #cpu_mem_after = get_process_memory_mb()                    # Perf measure
    peak_gpu_mb = get_peak_gpu_memory_mb()                     # Perf measure
    profile = fill_profile(train_time, inference_time, train_idx, valid_idx, 
                           #cpu_mem_after - cpu_mem_before, 
                           peak_gpu_mb, peak_gpu_mb,  
                           count_trainable_parameters(model), count_total_parameters(model)) # Perf measure
    train_df = pd.DataFrame(train_patient_embeddings, index=train_idx)
    test_df = pd.DataFrame(test_patient_embeddings, index=valid_idx)
    train_df.columns = [f"{name}_{i}" for i in range(train_df.shape[1])]
    test_df.columns = [f"{name}_{i}" for i in range(test_df.shape[1])]
    return train_df, test_df, profile

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

    reset_gpu_memory()                          # Perf measure
    #cpu_mem_before = get_process_memory_mb()    # Perf measure
    profiler = MemoryProfiler(interval=0.01)
    profiler.start()
    t0 = time.perf_counter()                    # Perf measure
    model = CEHRBERTModel(vocab_size=len(word_to_idx),embed_dim=embed_size, hidden_dim=hidden_size,
                          num_heads=num_heads,num_layers=num_layers,max_len=max_len,pooling=pooling,
                          use_token_type=use_token_type,name=name)
    print(model)
    model.train_model(train_dataloader,val_loader=val_dataloader,num_epochs=num_epochs, lr=lr,
                    loss_type=loss_type,alpha=alpha,gamma=gamma,weight_decay=weight_decay,
                    enable_plot=enable_plot,frame_tqdm=frame_tqdm,frame_plot=frame_plot, plotsize=plotsize)
    if torch.cuda.is_available(): torch.cuda.synchronize()     # Perf measure
    train_time = time.perf_counter() - t0                      # Perf measure
    if torch.cuda.is_available(): torch.cuda.synchronize()     # Perf measure

    t0 = time.perf_counter()     # Perf measure
    train_embeddings, train_ids = model.get_embeddings(full_train_dataloader)
    valid_embeddings, valid_ids = model.get_embeddings(test_dataloader)
    if torch.cuda.is_available(): torch.cuda.synchronize()     # Perf measure
    inference_time = time.perf_counter() - t0                  # Perf measure
    ##cpu_mem_after = get_process_memory_mb()                    # Perf measure
    memory_stats = profiler.stop()
    peak_memory_mb = memory_stats["peak_rss_mb"]
    peak_gpu_mb = get_peak_gpu_memory_mb()                     # Perf measure
    profile = fill_profile(train_time, inference_time, train_idx, valid_idx, 
                           #cpu_mem_after - cpu_mem_before, 
                           peak_memory_mb, peak_gpu_mb,  
                           count_trainable_parameters(model), count_total_parameters(model) ) # Perf measure

    train_df = pd.DataFrame(train_embeddings, index=train_ids)
    valid_df = pd.DataFrame(valid_embeddings, index=valid_ids)

    train_df.columns = [f"{name}_{i}" for i in range(train_df.shape[1])]
    valid_df.columns = [f"{name}_{i}" for i in range(valid_df.shape[1])]

    return train_df, valid_df, profile

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
    t0 = time.perf_counter()                    # Perf measure
    #cpu_mem_before = get_process_memory_mb()    # Perf measure
    profiler = MemoryProfiler(interval=0.01)
    profiler.start()
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
    model = DOME()
    W, C_plus, C_minus = model.fit(P_plus, P_minus)
    train_time = time.perf_counter() - t0                      # Perf measure

    vocab_lower = [w.lower() for w in cooc_prior.columns]
    W_df = pd.DataFrame(W.T, columns=vocab_lower)
    C_df = pd.DataFrame(C_minus.T, columns=vocab_lower)
    
    t0 = time.perf_counter()     # Perf measure
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
    inference_time = time.perf_counter() - t0                  # Perf measure
    #cpu_mem_after = get_process_memory_mb()

    memory_stats = profiler.stop()
    peak_memory_mb = memory_stats["peak_rss_mb"]
    #peak_increase_mb = memory_stats["peak_increase_mb"]
    profile = fill_profile(train_time, inference_time, train_idx, valid_idx, 
                           #cpu_mem_after - cpu_mem_before, 
                           peak_memory_mb,
                           np.nan, count_trainable_parameters(model), count_total_parameters(model) ) # Perf measure

    with frame_plot:
        print(f"✅ Embedding size {len(Xembed_dome.columns)}")

    train_df = pd.DataFrame(Xembed_dome.loc[train_idx], index=train_idx)
    test_df = pd.DataFrame(Xembed_dome.loc[valid_idx], index=valid_idx)
    train_df = rename_cols(train_df)
    test_df = rename_cols(test_df)
    return train_df, test_df, profile

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
    #new_names = {col: re.sub(r'[^A-Za-z0-9_]+', '', col) for col in df.columns}
    new_names = {
        col: re.sub(
            r'[^A-Za-z0-9_]+', '',
            re.sub(r'[\s,]+', '_', col)
        )
        for col in df.columns
    }
    new_n_list = list(new_names.values())
    # [LightGBM] Feature appears more than one time.
    new_names = {col: f'{new_col}_{i}' if new_col in new_n_list[:i] else new_col for i, (col, new_col) in enumerate(new_names.items())}
    return df.rename(columns=new_names)
    
def BINARYEmbedder(sequences,
                   train_idx=None,
                   valid_idx=None,
                   enable_plot=False,
                   frame_tqdm=None,
                   frame_plot=None,
                    ):
    #cpu_mem_before = get_process_memory_mb()    # Perf measure
    profiler = MemoryProfiler(interval=0.01)
    profiler.start()
    t0 = time.perf_counter()                    # Perf measure
    df_cnt = extract_event_counters(sequences)
    df_cnt = rename_cols(df_cnt)
    train_time = time.perf_counter() - t0                      # Perf measure
    t0 = time.perf_counter()     # Perf measure
    df_cnt[df_cnt > 1] = 1
    train_df = df_cnt.loc[train_idx]
    test_df = df_cnt.loc[valid_idx]
    inference_time = time.perf_counter() - t0                  # Perf measure
    #cpu_mem_after = get_process_memory_mb()                    # Perf measure
    memory_stats = profiler.stop()
    peak_memory_mb = memory_stats["peak_rss_mb"]
    #peak_increase_mb = memory_stats["peak_increase_mb"]
    profile = fill_profile(train_time, inference_time, train_idx, valid_idx, 
                           #cpu_mem_after - cpu_mem_before, 
                           peak_memory_mb,
                           np.nan, len(df_cnt.columns), np.nan ) # Perf measure
    if enable_plot:
        print(f"Counting event names: {' '.join(list(df_cnt.columns))}")
    return train_df, test_df, profile

def FREQEmbedder(sequences,
                   train_idx=None,
                   valid_idx=None,
                   enable_plot=False,
                   frame_tqdm=None,
                   frame_plot=None,
                    ):
    #cpu_mem_before = get_process_memory_mb()    # Perf measure
    profiler = MemoryProfiler(interval=0.01)
    profiler.start()
    t0 = time.perf_counter()                    # Perf measure
    df_cnt = extract_event_counters(sequences)
    df_cnt = rename_cols(df_cnt)
    train_time = time.perf_counter() - t0                      # Perf measure
    t0 = time.perf_counter()     # Perf measure
    train_df = df_cnt.loc[train_idx]
    test_df = df_cnt.loc[valid_idx]
    inference_time = time.perf_counter() - t0                  # Perf measure
    #cpu_mem_after = get_process_memory_mb()                    # Perf measure
    memory_stats = profiler.stop()
    peak_memory_mb = memory_stats["peak_rss_mb"]
    #peak_increase_mb = memory_stats["peak_increase_mb"]
    profile = fill_profile(train_time, inference_time, train_idx, valid_idx, 
                           #cpu_mem_after - cpu_mem_before, 
                           peak_memory_mb, 
                           np.nan, len(df_cnt.columns), np.nan ) # Perf measure
    if enable_plot:
        print(f"Frequency event names: {' '.join(list(df_cnt.columns))}")
    return train_df, test_df, profile

def COUNTEmbedder(sequences,
                   train_idx=None,
                   valid_idx=None,
                   enable_plot=False,
                   frame_tqdm=None,
                   frame_plot=None,
                    ):
    
    #cpu_mem_before = get_process_memory_mb()    # Perf measure
    profiler = MemoryProfiler(interval=0.01)
    profiler.start()
    t0 = time.perf_counter()                    # Perf measure
    df_cnt = extract_event_type_counters(sequences)
    train_time = time.perf_counter() - t0                      # Perf measure
    t0 = time.perf_counter()     # Perf measure
    train_df = df_cnt.loc[train_idx]
    test_df = df_cnt.loc[valid_idx]
    inference_time = time.perf_counter() - t0                  # Perf measure
    #cpu_mem_after = get_process_memory_mb()                    # Perf measure
    memory_stats = profiler.stop()
    peak_memory_mb = memory_stats["peak_rss_mb"]
    #peak_increase_mb = memory_stats["peak_increase_mb"]
    profile = fill_profile(train_time, inference_time, train_idx, valid_idx, 
                           #cpu_mem_after - cpu_mem_before, 
                           peak_memory_mb,
                           np.nan, len(df_cnt.columns), np.nan, np.nan ) # Perf measure
    if enable_plot:
        print(f"Counting event types: {' '.join(list(df_cnt.columns))}")
    return train_df, test_df, profile

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
    print(model)
    reset_gpu_memory()                          # Perf measure
    #cpu_mem_before = get_process_memory_mb()    # Perf measure
    profiler = MemoryProfiler(interval=0.01)
    profiler.start()
    t0 = time.perf_counter()                    # Perf measure
    model.train_model(train_loader, val_loader=None, num_epochs=num_epochs, enable_plot=enable_plot, 
                      frame_tqdm=frame_tqdm, frame_plot=frame_plot, plotsize=plotsize)

    if torch.cuda.is_available(): torch.cuda.synchronize()     # Perf measure
    train_time = time.perf_counter() - t0                      # Perf measure
    if torch.cuda.is_available(): torch.cuda.synchronize()     # Perf measure
    t0 = time.perf_counter()     # Perf measure
    train_patient_embeddings, tids = model.get_embeddings(train_loader)
    test_patient_embeddings, tsids = model.get_embeddings(test_loader)
    if torch.cuda.is_available(): torch.cuda.synchronize()     # Perf measure
    inference_time = time.perf_counter() - t0                  # Perf measure
    #cpu_mem_after = get_process_memory_mb()                    # Perf measure
    memory_stats = profiler.stop()
    peak_memory_mb = memory_stats["peak_rss_mb"]
    #peak_increase_mb = memory_stats["peak_increase_mb"]
    peak_gpu_mb = get_peak_gpu_memory_mb()                     # Perf measure
    profile = fill_profile(train_time, inference_time, train_idx, valid_idx, 
                           #cpu_mem_after - cpu_mem_before, 
                           peak_memory_mb, peak_gpu_mb,  
                           count_trainable_parameters(model), count_total_parameters(model) ) # Perf measure

    train_df = pd.DataFrame(train_patient_embeddings, index=train_idx)
    test_df = pd.DataFrame(test_patient_embeddings, index=valid_idx)
    train_df.columns = [f"gru_{i}" for i in range(train_df.shape[1])]
    test_df.columns = [f"gru_{i}" for i in range(test_df.shape[1])]
    return train_df, test_df, profile

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
    print(model)
    
    reset_gpu_memory()                          # Perf measure
    #cpu_mem_before = get_process_memory_mb()    # Perf measure
    profiler = MemoryProfiler(interval=0.01)
    profiler.start()
    t0 = time.perf_counter()                    # Perf measure
    model.train_model(train_loader, val_loader=None, num_epochs=num_epochs, enable_plot=enable_plot, 
                      frame_tqdm=frame_tqdm, frame_plot=frame_plot, plotsize=plotsize)
    
    if torch.cuda.is_available(): torch.cuda.synchronize()     # Perf measure
    train_time = time.perf_counter() - t0                      # Perf measure
    if torch.cuda.is_available(): torch.cuda.synchronize()     # Perf measure
    t0 = time.perf_counter()     # Perf measure
    train_patient_embeddings, tids = model.get_embeddings(train_loader)
    test_patient_embeddings, tsids = model.get_embeddings(test_loader)
    if torch.cuda.is_available(): torch.cuda.synchronize()     # Perf measure
    inference_time = time.perf_counter() - t0                  # Perf measure
    #cpu_mem_after = get_process_memory_mb()                    # Perf measure
    memory_stats = profiler.stop()
    peak_memory_mb = memory_stats["peak_rss_mb"]
    #peak_increase_mb = memory_stats["peak_increase_mb"]
    peak_gpu_mb = get_peak_gpu_memory_mb()                     # Perf measure
    profile = fill_profile(train_time, inference_time, train_idx, valid_idx, 
                           #cpu_mem_after - cpu_mem_before, 
                           peak_memory_mb, peak_gpu_mb,  
                           count_trainable_parameters(model), count_total_parameters(model) ) # Perf measure

    train_df = pd.DataFrame(train_patient_embeddings, index=train_idx)
    test_df = pd.DataFrame(test_patient_embeddings, index=valid_idx)
    train_df.columns = [f"dipole_{i}" for i in range(train_df.shape[1])]
    test_df.columns = [f"dipole_{i}" for i in range(test_df.shape[1])]
    return train_df, test_df, profile

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
    reset_gpu_memory()                          # Perf measure
    #cpu_mem_before = get_process_memory_mb()    # Perf measure
    profiler = MemoryProfiler(interval=0.01)
    profiler.start()
    t0 = time.perf_counter()                    # Perf measure
    model = Med2VecModel(vocab_size=len(word_to_idx),embed_dim=embed_size, visit_dim=hidden_size)
    print(model)
    model.train_model(train_loader, val_loader=None, num_epochs=num_epochs, lambda_visit=lambda_visit, enable_plot=enable_plot, 
                      frame_tqdm=frame_tqdm, frame_plot=frame_plot, plotsize=plotsize)
    if torch.cuda.is_available(): torch.cuda.synchronize()     # Perf measure
    train_time = time.perf_counter() - t0                      # Perf measure
    if torch.cuda.is_available(): torch.cuda.synchronize()     # Perf measure
    t0 = time.perf_counter()     # Perf measure
    train_emb, tids = model.get_embeddings(train_loader)
    valid_emb, vids = model.get_embeddings(valid_loader)
    if torch.cuda.is_available(): torch.cuda.synchronize()     # Perf measure
    inference_time = time.perf_counter() - t0                  # Perf measure
    #cpu_mem_after = get_process_memory_mb()                    # Perf measure
    memory_stats = profiler.stop()
    peak_memory_mb = memory_stats["peak_rss_mb"]
    #peak_increase_mb = memory_stats["peak_increase_mb"]
    peak_gpu_mb = get_peak_gpu_memory_mb()                     # Perf measure
    profile = fill_profile(train_time, inference_time, train_idx, valid_idx, 
                           #cpu_mem_after - cpu_mem_before, 
                           peak_memory_mb, peak_gpu_mb,  
                           count_trainable_parameters(model), count_total_parameters(model) ) # Perf measure

    df_train = pd.DataFrame(train_emb, index=train_idx)
    df_valid = pd.DataFrame(valid_emb, index=valid_idx)
    df_train.columns = [f"M2V_{i}" for i in range(df_train.shape[1])]
    df_valid.columns = [f"M2V_{i}" for i in range(df_valid.shape[1])]

    return df_train, df_valid, profile

def sequences_to_tfidf_tokens(
    sequences,
    patient_ids
):
    """
    sequences:  dict[id] = [ (event, date, type), ...]
                dict[id] = [ (event, date), ...]
    """
    docs = []
    for pid in patient_ids:
        patient_events = sequences[pid]
        tokens = []
        for item in patient_events:
            event = item[0]
            if isinstance(event, str):
                tokens.append(event)
            else:
                # nel caso event sia lista/tuple
                tokens.extend(
                    [str(e) for e in event]
                )
        docs.append(tokens)
    return docs

def TFIDFembedder(sequences, train_idx, valid_idx,labels,
            embedding_dim=128, max_features=None, min_df=2, max_df=1.0,
            ngram_range=(1, 1), use_svd=True, sublinear_tf=True, random_state=42, enable_plot=False,
            frame_tqdm=None,
            frame_plot=None,
            name="TFIDF"
):

    # ==================================================
    # PREPARA LE SEQUENZE
    # ==================================================

    train_docs = sequences_to_tfidf_tokens(sequences,train_idx)
    valid_docs = sequences_to_tfidf_tokens(sequences,valid_idx)

    # ==================================================
    # LABEL
    # ==================================================

    y_train = np.array([labels[i]for i in train_idx])
    y_valid = np.array([labels[i]for i in valid_idx])

    # ==================================================
    # MODEL
    # ==================================================

    model = TFIDFModel(max_features=max_features, min_df=min_df, max_df=max_df,
        ngram_range=ngram_range,sublinear_tf=sublinear_tf,use_svd=use_svd,embedding_dim=embedding_dim,
        random_state=random_state,name=name
    )
    print(model)

    # ==================================================
    # TRAIN
    # ==================================================

    reset_gpu_memory()                          # Perf measure
    #cpu_mem_before = get_process_memory_mb()    # Perf measure
    profiler = MemoryProfiler(interval=0.01)
    profiler.start()
    t0 = time.perf_counter()                    # Perf measure
    model.train_model(train_sequences=train_docs,train_labels=y_train,
        val_sequences=valid_docs,val_labels=y_valid,
        enable_plot=enable_plot,frame_tqdm=frame_tqdm,frame_plot=frame_plot
    )
    if torch.cuda.is_available(): torch.cuda.synchronize()     # Perf measure
    train_time = time.perf_counter() - t0                      # Perf measure
    if torch.cuda.is_available(): torch.cuda.synchronize()     # Perf measure

    # ==================================================
    # EMBEDDINGS
    # ==================================================

    t0 = time.perf_counter()     # Perf measure
    train_embeddings, train_ids = model.get_embeddings(train_docs,ids=train_idx)
    valid_embeddings, valid_ids = model.get_embeddings(valid_docs,ids=valid_idx)
    if torch.cuda.is_available(): torch.cuda.synchronize()     # Perf measure
    inference_time = time.perf_counter() - t0                  # Perf measure
    #cpu_mem_after = get_process_memory_mb()                    # Perf measure
    memory_stats = profiler.stop()
    peak_memory_mb = memory_stats["peak_rss_mb"]
    #peak_increase_mb = memory_stats["peak_increase_mb"]
    peak_gpu_mb = get_peak_gpu_memory_mb()                     # Perf measure
    profile = fill_profile(train_time, inference_time, train_idx, valid_idx, 
                           #cpu_mem_after - cpu_mem_before, 
                           peak_memory_mb, peak_gpu_mb,  
                           count_trainable_parameters(model), count_total_parameters(model) ) # Perf measure

    # ==================================================
    # DATAFRAME
    # ==================================================

    train_df = pd.DataFrame(train_embeddings,index=train_ids)
    valid_df = pd.DataFrame(valid_embeddings,index=valid_ids)

    columns = [f"{name}_{i}" for i in range(train_df.shape[1])]

    train_df.columns = columns
    valid_df.columns = columns

    return train_df, valid_df, profile

def sequences_to_bow_tokens(
    sequences,
    patient_ids
):
    docs = []
    for pid in patient_ids:
        patient_events = sequences[pid]
        tokens = []
        for item in patient_events:
            event = item[0]
            if isinstance(event, str):
                tokens.append(event)
            else:
                tokens.extend(
                    [str(e) for e in event]
                )
        docs.append(tokens)
    return docs


def BOWembedder(
    sequences,
    labels,
    train_idx,
    valid_idx,
    embedding_dim=128,
    max_features=None,
    min_df=2,
    max_df=1.0,
    ngram_range=(1, 1),
    binary=False,
    use_svd=True,
    random_state=42,
    enable_plot=False,
    frame_tqdm=None,
    frame_plot=None,
    name="BOW"
):

    train_docs = sequences_to_bow_tokens(
        sequences,
        train_idx
    )

    valid_docs = sequences_to_bow_tokens(
        sequences,
        valid_idx
    )

    y_train = np.array([
        labels[i]
        for i in train_idx
    ])

    y_valid = np.array([
        labels[i]
        for i in valid_idx
    ])

    reset_gpu_memory()                          # Perf measure
    #cpu_mem_before = get_process_memory_mb()    # Perf measure
    profiler = MemoryProfiler(interval=0.01)
    profiler.start()
    t0 = time.perf_counter()                    # Perf measure
    model = BOWModel(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        binary=binary,
        use_svd=use_svd,
        embedding_dim=embedding_dim,
        random_state=random_state,
        name=name
    )
    print(model)

    model.train_model(
        train_sequences=train_docs,
        train_labels=y_train,
        val_sequences=valid_docs,
        val_labels=y_valid,
        enable_plot=enable_plot,
        frame_tqdm=frame_tqdm,
        frame_plot=frame_plot
    )
    if torch.cuda.is_available(): torch.cuda.synchronize()     # Perf measure
    train_time = time.perf_counter() - t0                      # Perf measure
    if torch.cuda.is_available(): torch.cuda.synchronize()     # Perf measure

    t0 = time.perf_counter()     # Perf measure
    train_embeddings, train_ids = model.get_embeddings(
        train_docs,
        ids=train_idx
    )

    valid_embeddings, valid_ids = model.get_embeddings(
        valid_docs,
        ids=valid_idx
    )
    if torch.cuda.is_available(): torch.cuda.synchronize()     # Perf measure
    inference_time = time.perf_counter() - t0                  # Perf measure
    #cpu_mem_after = get_process_memory_mb()                    # Perf measure
    memory_stats = profiler.stop()
    peak_memory_mb = memory_stats["peak_rss_mb"]
    #peak_increase_mb = memory_stats["peak_increase_mb"]
    peak_gpu_mb = get_peak_gpu_memory_mb()                     # Perf measure
    profile = fill_profile(train_time, inference_time, train_idx, valid_idx, 
                           #cpu_mem_after - cpu_mem_before, 
                           peak_memory_mb, peak_gpu_mb,  
                           count_trainable_parameters(model), count_total_parameters(model) ) # Perf measure

    train_df = pd.DataFrame(
        train_embeddings,
        index=train_ids
    )

    valid_df = pd.DataFrame(
        valid_embeddings,
        index=valid_ids
    )

    columns = [
        f"{name}_{i}"
        for i in range(train_df.shape[1])
    ]

    train_df.columns = columns
    valid_df.columns = columns

    return train_df, valid_df, profile