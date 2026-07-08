import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from IPython.display import clear_output, display, update_display
from scipy.sparse import coo_matrix
from collections import defaultdict
import time
import pandas as pd
from tqdm.notebook import tqdm
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import math
from sklearn.metrics import (
    matthews_corrcoef, confusion_matrix, accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score
)
from contextlib import nullcontext

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: output grezzo del modello, shape [batch] o [batch, 1]
        targets: 0/1, shape [batch] o [batch, 1]
        """

        logits = logits.view(-1)
        targets = targets.float().view(-1)

        bce_loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none"
        )

        probas = torch.sigmoid(logits)

        p_t = probas * targets + (1 - probas) * (1 - targets)

        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        focal_factor = (1 - p_t) ** self.gamma

        loss = alpha_t * focal_factor * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
            
#------------------------------------------------------------------------------------------------
# Plot function
#------------------------------------------------------------------------------------------------
def plot_foo(name, t_losses, v_losses, size=(7, 5)):
    # aggiorna SOLO il frame del grafico
    clear_output(wait=True)
    
    plt.figure(figsize=size)
    plt.plot(t_losses, label="Train Loss")
    
    if len(v_losses) > 0:
        plt.title(f"[{name}] Training & Validation Loss Curve")
        plt.plot(v_losses, label="Val Loss")
    else:
        plt.title(f"[{name}] Training Loss Curve")
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()


#------------------------------------------------------------------------------------------------
# DOME
#------------------------------------------------------------------------------------------------
def riskmatrix_loop_fb_dome(events, attributes, targets, emb1, emb2):
    attributes = [a.lower() for a in attributes]
    targets = [t.lower() for t in targets]
    zero_data = np.ones((len(events), len(attributes)))
    X_df = pd.DataFrame(zero_data, columns=attributes, dtype=float, index=events.keys())

    Wmul = emb1.T@emb2
    for id, dcount in tqdm(events.items(), desc="Risk calculating"):
        for concept in dcount.keys():
            concept_l = concept.lower()
            if concept_l in attributes:
                risk_sum = 0.0
                for disease in targets:
                        w_val = Wmul[concept_l][disease]
                        risk_sum += 1 / float(1 + math.exp(w_val))
                X_df.loc[id, concept_l] += dcount[concept] * risk_sum
    return X_df

class DOME:
    def __init__(self, dim=128, lr=1e-3, max_iter=200, mu=0.9):
        self.dim = dim
        self.lr = lr
        self.max_iter = max_iter
        self.mu = mu

    def fit(self, P_plus, P_minus):
        n = P_plus.shape[0]
        W = np.random.normal(scale=0.01, size=(n, self.dim))
        C_plus = np.random.normal(scale=0.01, size=(self.dim, n))
        C_minus = np.random.normal(scale=0.01, size=(self.dim, n))

        W_m, Cp_m, Cm_m = np.zeros_like(W), np.zeros_like(C_plus), np.zeros_like(C_minus)

        for i in range(self.max_iter):
            # forward
            pred_p = W @ C_plus
            pred_m = W @ C_minus

            # gradients
            grad_W = (pred_p - P_plus) @ C_plus.T + (pred_m - P_minus) @ C_minus.T
            grad_Cp = W.T @ (pred_p - P_plus)
            grad_Cm = W.T @ (pred_m - P_minus)

            # momentum update
            W_m = self.mu * W_m - self.lr * grad_W
            Cp_m = self.mu * Cp_m - self.lr * grad_Cp
            Cm_m = self.mu * Cm_m - self.lr * grad_Cm

            W += W_m
            C_plus += Cp_m
            C_minus += Cm_m

        return W, C_plus.T, C_minus.T
    
def compute_directional_ppmi(cooc_matrix, k=1):
    total = np.sum(cooc_matrix)
    row_sum = np.sum(cooc_matrix, axis=1, keepdims=True)
    col_sum = np.sum(cooc_matrix, axis=0, keepdims=True)
    expected = row_sum @ col_sum / total
    expected[expected == 0] = 1e-10
    ppmi = np.log((cooc_matrix * total) / expected)
    return np.maximum(ppmi - np.log(k), 0)

def co_occurrence_infectious_window(events, infectious_set, df_clinica, months_window=5, direction='prior', skiptoken='<ph>', exclude_from_rows=None, disable=False):
    from collections import defaultdict
    import pandas as pd
    import numpy as np

    d = defaultdict(int)
    vocabular = set()

    if exclude_from_rows is None:
        exclude_from_rows = set()

    for pid, timeline in tqdm(events.items(), desc="Building co-occurrence", disable=False):
        timeline_sorted = sorted(timeline, key=lambda x: x[1])
        flat_sequence = [(ev.lower(), pd.to_datetime(date)) for ev, date in timeline_sorted if isinstance(ev, str) and pd.notna(date)]
        if not flat_sequence:
            continue

        infectious_dates = [date for ev, date in flat_sequence if ev in infectious_set]

        if infectious_dates:
            window_center = min(infectious_dates)
        else:
            followup_dates = [date for ev, date in flat_sequence]
            #followup_dates = [date for ev, date in flat_sequence if ev == 'followup']
            #if not followup_dates:
            #    try:
            #        raw = df_clinica.loc[id, 'date_flwup']
            #        if pd.notna(raw):
            #            dlist = [pd.to_datetime(d.strip(), errors='coerce') for d in str(raw).split(',')]
            #            dlist = [d for d in dlist if pd.notna(d)]
            #            if dlist:
            #                followup_dates = [max(dlist)]
            #    except Exception as e:
            #        print(f"⚠️ Errore nel recupero data estremale {id}: {e}")
            if not followup_dates:
                continue
            window_center = max(followup_dates)

        start_date = window_center - pd.DateOffset(months=months_window)
        end_date = window_center + pd.DateOffset(months=months_window)

        window_events = [(ev, date) for ev, date in flat_sequence if start_date <= date <= end_date and ev != skiptoken]

        for i, (token_i, date_i) in enumerate(window_events):
            if token_i in exclude_from_rows:
                continue
            vocabular.add(token_i)

            for j, (token_j, date_j) in enumerate(window_events):
                if i == j or token_j == skiptoken:
                    continue
                if direction == 'prior' and date_j >= date_i:
                    continue
                if direction == 'posterior' and date_j <= date_i:
                    continue
                d[(token_i, token_j)] += 1

    vocabular = sorted(vocabular)
    df = pd.DataFrame(0, index=vocabular, columns=vocabular, dtype=np.int32)
    for (w, c), val in d.items():
        if w in df.index and c in df.columns:
            df.at[w, c] = val

    return df, vocabular

#------------------------------------------------------------------------------------------------
# LSTM
#------------------------------------------------------------------------------------------------
def lstm_collate_fn(batch):
    ids = []
    sequences = []
    labels = []

    for item in batch:
        if len(item) == 2:
            _id, x = item
        elif len(item) == 3:
            _id, x, y = item
            labels.append(y)
        else:
            raise ValueError("Batch item must be (id,x) or (id,x,y)")

        ids.append(_id)
        sequences.append(x)

    padded_x = pad_sequence(sequences, batch_first=True, padding_value=0)

    if labels:
        y = torch.stack(labels).float()  # <-- tolto unsqueeze
        return ids, padded_x, y
    else:
        return ids, padded_x

    
class LSTMDataset(Dataset):
    def __init__(self, sequences_dict, labels_dict=None):
        """
        sequences_dict: dict {id: sequence} 
        labels_dict: dict {id: label}, opzionale
        """
        #self.ids = sorted(sequences_dict.keys())  # ordine deterministico
        self.ids = list(sequences_dict.keys())
        self.sequences = [torch.tensor(sequences_dict[_id], dtype=torch.long) for _id in self.ids]
        if labels_dict is not None:
            self.labels = [torch.tensor(labels_dict[_id], dtype=torch.float) for _id in self.ids]
        else:
            self.labels = None

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x = self.sequences[idx]
        _id = self.ids[idx]
        if self.labels is None:
            return _id, x
        else:
            y = self.labels[idx]
            return _id, x, y
        
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pooling=False, name='LSTM'):
        super().__init__()
        self.name = name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pooling = pooling
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hn, cn) = self.lstm(embedded)
        feat = output.mean(dim=1) if self.pooling else hn.squeeze(0)
        logits = self.classifier(feat)
        return logits, feat

    def train_model(self, 
                    train_loader, 
                    val_loader=None, 
                    num_epochs=10, 
                    lr=1e-3,
                    enable_plot=False,
                    frame_tqdm=None,
                    frame_plot=None,
                    plotsize=(4,5)):
        
        self.to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        train_losses, val_losses, val_aucs = [], [], []

        with frame_tqdm:
            frame_tqdm.clear_output(wait=True)
            pbar = tqdm(range(num_epochs), total=num_epochs, desc=f'Training [{self.name}]')
            for epoch in pbar:
                # TRAINING
                self.train()
                total_loss = 0

                for batch in train_loader:
                    optimizer.zero_grad()
                    if len(batch) == 3:
                        ids, x, y = batch
                        y = y.float().unsqueeze(1).to(self.device)
                    elif len(batch) == 2:
                        ids, x = batch
                        y = None
                    else:
                        raise ValueError("Batch deve essere (id,x) o (id,x,y)")

                    x = x.long().to(self.device)
                    logits, _ = self(x)

                    if y is not None:
                        loss = criterion(logits, y)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()

                avg_train_loss = total_loss / len(train_loader) if total_loss > 0 else 0
                train_losses.append(avg_train_loss)

                # VALIDATION
                if val_loader is not None:
                    val_loss, auc = self.evaluate(val_loader)
                    val_losses.append(val_loss)
                    val_aucs.append(auc)
                    
                # PROGRESS BAR
                if val_loader:
                    pbar.set_postfix({'Train Loss': round(avg_train_loss,3), 'Val Loss': round(val_loss,3), 'AUC': round(auc,3)})
                else:
                    pbar.set_postfix({'Train Loss': round(avg_train_loss,3)})

                # OPTIONAL PLOT
                with frame_plot:
                    plot_foo(self.name, train_losses, val_losses, size=plotsize) if enable_plot else None

        return train_losses, val_losses

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.eval()
        criterion = nn.BCEWithLogitsLoss()
        total_loss = 0
        preds, trues = [], []

        for batch in dataloader:
            if len(batch) == 3:
                ids, x, y = batch
                y = y.float().unsqueeze(1).to(self.device)
            elif len(batch) == 2:
                ids, x = batch
                y = None
            else:
                raise ValueError("Batch deve essere (id,x) o (id,x,y)")

            x = x.long().to(self.device)
            logits, _ = self(x)

            if y is not None:
                loss = criterion(logits, y)
                total_loss += loss.item()
                probs = torch.sigmoid(logits).cpu().numpy()
                preds.extend(probs)
                trues.extend(y.cpu().numpy())

        avg_loss = total_loss / len(dataloader) if total_loss > 0 else 0
        try:
            auc = roc_auc_score(trues, preds) if len(trues) > 0 else float("nan")
        except:
            auc = float("nan")

        return avg_loss, auc
    
    @torch.no_grad()
    def get_embeddings(self, dataloader):
        """
        Estrae gli embedding LSTM per ogni batch del dataloader.

        Ritorna:
            embeddings: np.array di shape (N, hidden_dim)
            ids: np.array di shape (N,)
        """
        self.eval()
        all_feats = []
        all_ids = []

        for batch in dataloader:
            if len(batch) >= 2:
                ids, x = batch[:2]
            else:
                raise ValueError("Batch deve avere almeno (id, x)")

            x = x.long().to(self.device)
            _, feats = self(x)  # (batch, hidden_dim)

            all_feats.append(feats.cpu().numpy())
            all_ids.extend(ids)

        embeddings = np.vstack(all_feats)
        ids_array = np.array(all_ids)

        return embeddings, ids_array

def bipadlstm_collate_fn(batch):

    ids = []
    sequences = []
    labels = []
    has_labels = len(batch[0]) == 3
    for item in batch:
        if has_labels:
            i, seq, y = item
            labels.append(y)
        else:
            i, seq = item
        ids.append(i)
        sequences.append(torch.tensor(seq))
    lengths = torch.tensor([len(s) for s in sequences])
    sequences = nn.utils.rnn.pad_sequence(
        sequences,
        batch_first=True,
        padding_value=0
    )
    if has_labels:
        labels = torch.tensor(labels)

#------------------------------------------------------------------------------------------------
# Flexible LSTM wih optional padding, bidirectional and attention
#------------------------------------------------------------------------------------------------
class FlexibleLSTMModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_dim,
        pooling=False,
        bidirectional=False,
        use_padding=False,
        use_attention=False,
        name="FlexLSTM"
    ):
        super().__init__()
        self.name = name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pooling = pooling
        self.bidirectional = bidirectional
        self.use_padding = use_padding
        self.use_attention = use_attention
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=bidirectional
        )
        feature_dim = hidden_dim * self.num_directions
        if use_attention:
            self.attention = nn.Linear(feature_dim, 1)
        self.classifier = nn.Linear(feature_dim, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        # --------------------------------
        # LSTM (con o senza padding)
        # --------------------------------
        if self.use_padding:
            lengths = (x != 0).sum(dim=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False
            )
            packed_output, (hn, cn) = self.lstm(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=x.size(1)
            )
        else:
            output, (hn, cn) = self.lstm(embedded)
        # --------------------------------
        # FEATURE EXTRACTION
        # --------------------------------
        if self.use_attention:
            mask = (x != 0)
            scores = self.attention(output).squeeze(-1)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores, dim=1)
            feat = torch.sum(output * weights.unsqueeze(-1), dim=1)
        else:
            if self.pooling:
                feat = output.mean(dim=1)
            else:
                if self.bidirectional:
                    h_forward = hn[0]
                    h_backward = hn[1]
                    feat = torch.cat((h_forward, h_backward), dim=1)
                else:
                    feat = hn.squeeze(0)
        logits = self.classifier(feat)
        return logits, feat

    def train_model(self, 
                    train_loader, 
                    val_loader=None, 
                    num_epochs=10, 
                    lr=1e-3,
                    enable_plot=False,
                    frame_tqdm=None,
                    frame_plot=None,
                    plotsize=(4,5)):
        
        self.to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()
        #criterion = BinaryFocalLoss(alpha=0.75, gamma=2.0)
        train_losses, val_losses, val_aucs = [], [], []
        with frame_tqdm:
            frame_tqdm.clear_output(wait=True)
            pbar = tqdm(range(num_epochs), total=num_epochs, desc=f'Training [{self.name}]')
            for epoch in pbar:
                self.train()
                total_loss = 0
                for batch in train_loader:
                    optimizer.zero_grad()
                    if len(batch) == 3:
                        ids, x, y = batch
                        y = y.float().unsqueeze(1).to(self.device)
                    elif len(batch) == 2:
                        ids, x = batch
                        y = None
                    else:
                        raise ValueError("Batch deve essere (id,x) o (id,x,y)")
                    x = x.long().to(self.device)
                    logits, _ = self(x)
                    if y is not None:
                        loss = criterion(logits, y)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                avg_train_loss = total_loss / len(train_loader) if total_loss > 0 else 0
                train_losses.append(avg_train_loss)

                if val_loader is not None:
                    val_loss, auc = self.evaluate(val_loader)
                    val_losses.append(val_loss)
                    val_aucs.append(auc)
                if val_loader:
                    pbar.set_postfix({
                        'Train Loss': round(avg_train_loss,3),
                        'Val Loss': round(val_loss,3),
                        'AUC': round(auc,3)
                    })
                else:
                    pbar.set_postfix({'Train Loss': round(avg_train_loss,3)})

                with frame_plot:
                    plot_foo(self.name, train_losses, val_losses, size=plotsize) if enable_plot else None
        return train_losses, val_losses

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.eval()
        criterion = nn.BCEWithLogitsLoss()
        total_loss = 0
        preds, trues = [], []
        for batch in dataloader:
            if len(batch) == 3:
                ids, x, y = batch
                y = y.float().unsqueeze(1).to(self.device)
            elif len(batch) == 2:
                ids, x = batch
                y = None
            else:
                raise ValueError("Batch deve essere (id,x) o (id,x,y)")
            x = x.long().to(self.device)
            logits, _ = self(x)
            if y is not None:
                loss = criterion(logits, y)
                total_loss += loss.item()
                probs = torch.sigmoid(logits).cpu().numpy()
                preds.extend(probs)
                trues.extend(y.cpu().numpy())
        avg_loss = total_loss / len(dataloader) if total_loss > 0 else 0
        try:
            auc = roc_auc_score(trues, preds) if len(trues) > 0 else float("nan")
        except:
            auc = float("nan")
        return avg_loss, auc

    @torch.no_grad()
    def get_embeddings(self, dataloader):
        self.eval()
        all_feats = []
        all_ids = []
        for batch in dataloader:
            if len(batch) >= 2:
                ids, x = batch[:2]
            else:
                raise ValueError("Batch deve avere almeno (id, x)")
            x = x.long().to(self.device)
            _, feats = self(x)
            all_feats.append(feats.cpu().numpy())
            all_ids.extend(ids)
        embeddings = np.vstack(all_feats)
        ids_array = np.array(all_ids)
        return embeddings, ids_array
    
    @torch.no_grad()
    def predict_proba(self, dataloader):
        self.eval()

        all_probs = []

        for batch in dataloader:
            if len(batch) == 3:
                ids, x, _ = batch
            elif len(batch) == 2:
                ids, x = batch
            else:
                raise ValueError("Batch deve essere (id,x) o (id,x,y)")

            x = x.long().to(self.device)

            logits, _ = self(x)

            # 🔹 Binary case (BCEWithLogitsLoss)
            if logits.shape[1] == 1:
                prob_pos = torch.sigmoid(logits)                  # (batch, 1)
                probs = torch.cat([1 - prob_pos, prob_pos], dim=1)  # (batch, 2)

            # 🔹 Multiclass case (se mai lo userai)
            else:
                probs = torch.softmax(logits, dim=1)              # (batch, C)

            all_probs.append(probs.cpu())

        return torch.cat(all_probs, dim=0)

    @torch.no_grad()
    def predict(self, dataloader):
        probs = self.predict_proba(dataloader)
        preds = torch.argmax(probs, dim=1)
        return preds, probs   
       
#------------------------------------------------------------------------------------------------
# RETAIN
#------------------------------------------------------------------------------------------------
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    

class RETAINDataset(Dataset):
    def __init__(self, seqs, labels, num_features, reverse=True, create_dummy=True, name='RETAIN'):
        """
        Args:
            seqs (list): list of patients (list) of visits (list) of codes (int)
            labels (list): list of labels (int)
            num_features (int): number of total features available
            reverse (bool): If true, reverse order of visits for RETAIN
            create_dummy (bool): If True, replace empty sequences with one dummy visit of zeros
        """
        self.name = name
        if len(seqs) != len(labels):
            raise ValueError("Seqs and Labels have different lengths")

        self.seqs = []
        self.labels = [int(l) for l in labels]
        self.num_features = num_features
        self.reverse = reverse
        self.create_dummy = create_dummy

        for seq, label in zip(seqs, labels):
            if self.reverse:
                sequence = list(reversed(seq))
            else:
                sequence = list(seq)

            # Se la sequenza è vuota -> crea una visita dummy (riga di zeri)
            if len(sequence) == 0:
                if self.create_dummy:
                    # coo_matrix con shape (1, num_features) e nessun valore -> riga di zeri
                    mat = coo_matrix(([], ([], [])), shape=(1, num_features), dtype=np.float32)
                else:
                    # lascia come matrice di 0 righe (attenzione: può causare problemi a monte)
                    mat = coo_matrix(([], ([], [])), shape=(0, num_features), dtype=np.float32)
            else:
                rows = []
                cols = []
                vals = []
                for i, visit in enumerate(sequence):
                    # visit è una lista di codici (int), se usi eventi con pesi adatta vals
                    for code in visit:
                        if code < num_features:
                            rows.append(i)
                            cols.append(code)
                            vals.append(1.0)
                if len(vals) == 0:
                    # tutte le visite erano vuote -> crea comunque righe (len(sequence) righe) tutte zero
                    mat = coo_matrix(([], ([], [])), shape=(len(sequence), num_features), dtype=np.float32)
                else:
                    mat = coo_matrix((np.array(vals, dtype=np.float32),
                                      (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
                                     shape=(len(sequence), num_features),
                                     dtype=np.float32)
            # convert to CSR for faster row-slicing in collate / model if needed
            self.seqs.append(mat.tocsr())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # ritorna (csr_matrix, label)
        return self.seqs[index], self.labels[index]


def visit_collate_fn(batch):
    """
    DataLoaderIter call - self.collate_fn([self.dataset[i] for i in indices])
    Thus, 'batch' is a list [(seq1, label1), (seq2, label2), ... , (seqN, labelN)]
    where N is minibatch size, seq is a SparseFloatTensor, and label is a LongTensor

    :returns
        seqs
        labels
        lengths
    """
    batch_seq, batch_label = zip(*batch)

    num_features = batch_seq[0].shape[1]
    seq_lengths = list(map(lambda patient_tensor: patient_tensor.shape[0], batch_seq))
    max_length = max(seq_lengths)

    sorted_indices, sorted_lengths = zip(*sorted(enumerate(seq_lengths), key=lambda x: x[1], reverse=True))
    sorted_padded_seqs = []
    sorted_labels = []

    for i in sorted_indices:
        length = batch_seq[i].shape[0]

        if length < max_length:
            padded = np.concatenate(
                (batch_seq[i].toarray(), np.zeros((max_length - length, num_features), dtype=np.float32)), axis=0)
        else:
            padded = batch_seq[i].toarray()

        sorted_padded_seqs.append(padded)
        sorted_labels.append(batch_label[i])

    seq_tensor = np.stack(sorted_padded_seqs, axis=0)
    label_tensor = torch.tensor(sorted_labels, dtype=torch.long)

    return torch.from_numpy(seq_tensor), label_tensor, list(sorted_lengths)
   
class RETAINModel(nn.Module):
    def __init__(self, dim_input, dim_emb=128, dropout_input=0.8, dropout_emb=0.5, dim_alpha=128, dim_beta=128,
                    dropout_context=0.5, dim_output=2, l2=0.0001, batch_first=True, name='RETAIN'):
        super(RETAINModel, self).__init__()
        self.name = name
        self.embed_size = dim_emb
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_first = batch_first
        self.embedding = nn.Sequential(
            nn.Dropout(p=dropout_input),
            nn.Linear(dim_input, dim_emb, bias=False),
            nn.Dropout(p=dropout_emb)
        )
        init.xavier_normal(self.embedding[1].weight)

        self.rnn_alpha = nn.GRU(input_size=dim_emb, hidden_size=dim_alpha, num_layers=1, batch_first=self.batch_first)

        self.alpha_fc = nn.Linear(in_features=dim_alpha, out_features=1)
        init.xavier_normal(self.alpha_fc.weight)
        self.alpha_fc.bias.data.zero_()

        self.rnn_beta = nn.GRU(input_size=dim_emb, hidden_size=dim_beta, num_layers=1, batch_first=self.batch_first)

        self.beta_fc = nn.Linear(in_features=dim_beta, out_features=dim_emb)
        init.xavier_normal(self.beta_fc.weight, gain=nn.init.calculate_gain('tanh'))
        self.beta_fc.bias.data.zero_()

        self.output = nn.Sequential(
            nn.Dropout(p=dropout_context),
            nn.Linear(in_features=dim_emb, out_features=dim_output)
        )
        init.xavier_normal(self.output[1].weight)
        self.output[1].bias.data.zero_()


    def forward(self, x, lengths):
        if self.batch_first:
            batch_size, max_len = x.size()[:2]
        else:
            max_len, batch_size = x.size()[:2]

        # emb -> batch_size X max_len X dim_emb
        emb = self.embedding(x)

        packed_input = pack_padded_sequence(emb, lengths, batch_first=self.batch_first)

        g, _ = self.rnn_alpha(packed_input)

        # alpha_unpacked -> batch_size X max_len X dim_alpha
        alpha_unpacked, _ = pad_packed_sequence(g, batch_first=self.batch_first)

        # mask -> batch_size X max_len X 1
        mask = Variable(torch.FloatTensor(
            [[1.0 if i < lengths[idx] else 0.0 for i in range(max_len)] for idx in range(batch_size)]).unsqueeze(2),
                        requires_grad=False)
        if next(self.parameters()).is_cuda:  # returns a boolean
            mask = mask.cuda()

        # e => batch_size X max_len X 1
        e = self.alpha_fc(alpha_unpacked)

        def masked_softmax(batch_tensor, mask):
            exp = torch.exp(batch_tensor)
            masked_exp = exp * mask
            sum_masked_exp = torch.sum(masked_exp, dim=1, keepdim=True)
            return masked_exp / sum_masked_exp

        # Alpha = batch_size X max_len X 1
        # alpha value for padded visits (zero) will be zero
        alpha = masked_softmax(e, mask)

        h, _ = self.rnn_beta(packed_input)

        # beta_unpacked -> batch_size X max_len X dim_beta
        beta_unpacked, _ = pad_packed_sequence(h, batch_first=self.batch_first)

        # Beta -> batch_size X max_len X dim_emb
        # beta for padded visits will be zero-vectors
        #beta = F.tanh(self.beta_fc(beta_unpacked) * mask)
        beta = torch.tanh(self.beta_fc(beta_unpacked) * mask)

        # context -> batch_size X (1) X dim_emb (squeezed)
        # Context up to i-th visit context_i = sum(alpha_j * beta_j * emb_j)
        # Vectorized sum
        context = torch.bmm(torch.transpose(alpha, 1, 2), beta * emb).squeeze(1)

        # without applying non-linearity
        logit = self.output(context)

        return logit, alpha, beta

    def train_model(self, 
                    train_loader, 
                    val_loader=None,
                    num_epochs=10, 
                    enable_plot=False, 
                    frame_tqdm=None,
                    frame_plot=None,plotsize=(4,3),
                    ):
        """
        Unsupervised training to stabilize embedding space (MSE to zero)
        """
        self.to(self.device)

        optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.95)
        criterion = nn.CrossEntropyLoss()

        train_losses = []
        val_losses = []

        train_loss = 0
        val_loss = 0

        with frame_tqdm:
            frame_tqdm.clear_output(wait=True)
            pbar = tqdm(range(num_epochs), total=num_epochs, desc=f'Training [{self.name}]')
            for epoch in pbar:
                # TRAINING
                self.train()
                losses = AverageMeter()
                train_labels = []
                train_outputs = []

                for batch in train_loader:
                    inputs, targets, lengths = batch

                    input_var  = inputs.to(self.device)
                    target_var = targets.long().to(self.device).view(-1)
                    output, alpha, beta = self(input_var, lengths)
                    loss = criterion(output, target_var)
                    assert not np.isnan(loss.item()), "Model diverged with loss = NaN"

                    # log per metriche
                    train_labels.append(targets)
                    train_outputs.append(F.softmax(output, dim=1).data)

                    # update loss
                    losses.update(loss.item(), inputs.size(0))

                    # backward
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                train_loss = losses.avg
                train_losses.append(train_loss)

                train_y_true = torch.cat(train_labels, 0).cpu().numpy()
                train_y_pred = torch.cat(train_outputs, 0).cpu().numpy()
                
                # VALIDATION
                if val_loader:
                    self.eval()
                    val_losses_meter = AverageMeter()
                    valid_labels = []
                    valid_outputs = []

                    with torch.no_grad():
                        for batch in val_loader:
                            inputs, targets, lengths = batch

                            input_var  = inputs.to(self.device)
                            target_var = targets.to(self.device).view(-1)

                            output, alpha, beta = self(input_var, lengths)
                            loss = criterion(output, target_var)
                            val_losses_meter.update(loss.item(), inputs.size(0))

                            valid_labels.append(targets)
                            valid_outputs.append(F.softmax(output, dim=1).data)

                    val_loss = val_losses_meter.avg
                    val_losses.append(val_loss)

                    # AUROC
                    try:
                        y_true = torch.cat(valid_labels).numpy()
                        y_pred = torch.cat(valid_outputs).numpy()
                        auc = roc_auc_score(y_true, y_pred)
                    except:
                        auc = float("nan")
                        
                # PROGRESS BAR
                if val_loader:
                    pbar.set_postfix({'Train Loss': round(train_loss,3), 'Val Loss': round(val_loss,3), 'AUC': round(auc,3)})
                else:
                    pbar.set_postfix({'Train Loss': round(train_loss,3)})

                # PLOT (opzional)
                with frame_plot:
                    plot_foo(self.name, train_losses, val_losses, size=plotsize) if enable_plot else None

        return train_losses

    def get_embeddings_input(self, x):
        """
        Return the patient-level embedding vector before the final classification layer.
        x: (batch, seq_len, dim_input)
        """
        # x → embedding
        e = self.embedding(x)   # (batch, seq_len, dim_emb)
        e_rev = torch.flip(e, [1])

        # ---- α attention ----
        g, _ = self.rnn_alpha(e_rev)  # (batch, seq_len, dim_alpha)
        alpha_logits = self.alpha_fc(g)  # (batch, seq_len, 1)
        alpha_weights = torch.softmax(alpha_logits, dim=1)  # (batch, seq_len, 1)

        # ---- β attention ----
        h, _ = self.rnn_beta(e_rev)  # (batch, seq_len, dim_beta)
        beta = torch.tanh(self.beta_fc(h))  # (batch, seq_len, dim_emb)

        # ---- patient representation ----
        c = torch.sum(alpha_weights * beta * e_rev, dim=1)  # (batch, dim_emb)

        return c

    def get_embeddings(self, loader):
        #self.eval()
        all_embeds = np.empty(shape=(0,self.embed_size))
        all_ids = []

        with torch.no_grad():
            for batch in loader:
                # Gestione batch con o senza IDs
                if len(batch) == 4:
                    inputs, targets, lengths, ids = batch
                elif len(batch) == 3:
                    inputs, targets, lengths = batch
                    ids = list(range(len(inputs)))
                else:
                    raise ValueError(f"Unexpected batch length: {len(inputs)}")

                inputs = inputs.to(self.device)
                embeds = self.get_embeddings_input(inputs)  # (batch, dim_emb)

                all_embeds = np.concatenate((all_embeds,embeds.cpu().numpy()), axis=0)
                #all_embeds.append(embeds.cpu())
                all_ids.extend(ids)

        # Concatena i batch in un unico array 2D
        #all_embeds = torch.cat(all_embeds, dim=0).numpy()
        return all_embeds, np.array(all_ids)
        
#------------------------------------------------------------------------------------------------
# CEHR-BERT
#------------------------------------------------------------------------------------------------

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.85, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        logits = logits.view(-1)
        targets = targets.float().view(-1)

        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        loss = alpha_t * ((1 - p_t) ** self.gamma) * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class CEHRBERTDataset(Dataset):
    def __init__(
        self,
        sequences,
        labels_dict=None,
        word_to_idx=None,
        event_type_dict=None,
        max_len=512,
        unk_idx=1
    ):
        self.ids = list(sequences.keys())
        self.sequences = sequences
        self.labels_dict = labels_dict
        self.word_to_idx = word_to_idx
        self.event_type_dict = event_type_dict
        self.max_len = max_len
        self.unk_idx = unk_idx

    def __len__(self):
        return len(self.ids)

    def _convert_token(self, token):
        if isinstance(token, int):
            return token

        if self.word_to_idx is None:
            raise ValueError("Found string tokens but word_to_idx is None.")

        return self.word_to_idx.get(token, self.unk_idx)

    def __getitem__(self, idx):
        patient_id = self.ids[idx]
        seq = self.sequences[patient_id]

        input_ids = []

        for item in seq:
            if isinstance(item, tuple):
                token = item[0]
            else:
                token = item

            input_ids.append(self._convert_token(token))

        input_ids = input_ids[:self.max_len]

        item = {
            "id": patient_id,
            "input_ids": input_ids,
        }

        if self.labels_dict is not None:
            item["label"] = self.labels_dict[patient_id]

        #if self.event_type_dict is not None:
        #    item["token_types"] = self.event_type_dict[patient_id][:self.max_len]
        if self.event_type_dict is not None:
            token_types = self.event_type_dict[patient_id]

            if isinstance(token_types, tuple):
                print("Bad token_types tuple:", patient_id, token_types)
                raise ValueError(
                    f"event_type_dict[{patient_id}] deve essere una lista di interi, "
                    f"ma è una tuple: {token_types}"
                )

            token_types = list(token_types)[:self.max_len]

            if len(token_types) != len(input_ids):
                raise ValueError(
                    f"Mismatch for patient {patient_id}: "
                    f"len(input_ids)={len(input_ids)}, "
                    f"len(token_types)={len(token_types)}"
                )

            item["token_types"] = token_types

        return item



def cehrbert_collate_fn(batch, pad_idx=0):
    ids = [item["id"] for item in batch]

    sequences = [item["input_ids"] for item in batch]
    max_len = max(len(seq) for seq in sequences)

    input_ids = []
    attention_mask = []

    for seq in sequences:
        seq = list(seq)
        pad_len = max_len - len(seq)

        input_ids.append(seq + [pad_idx] * pad_len)
        attention_mask.append([1] * len(seq) + [0] * pad_len)

    out = {
        "ids": ids,
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }

    if "label" in batch[0]:
        out["labels"] = torch.tensor(
            [item["label"] for item in batch],
            dtype=torch.float
        )

    if "token_types" in batch[0]:
        padded_types = []

        for item in batch:
            tt = item["token_types"]

            # caso errato ma frequente: (id, [types])
            if isinstance(tt, tuple):
                if len(tt) == 2 and isinstance(tt[1], (list, tuple)):
                    tt = tt[1]
                else:
                    raise ValueError(
                        f"token_types for id={item['id']} is a bad tuple: {tt}"
                    )

            tt = list(tt)

            if len(tt) != len(item["input_ids"]):
                raise ValueError(
                    f"Mismatch id={item['id']}: "
                    f"len(input_ids)={len(item['input_ids'])}, "
                    f"len(token_types)={len(tt)}"
                )

            pad_len = max_len - len(tt)
            padded_types.append(tt + [0] * pad_len)

        out["token_types"] = torch.tensor(padded_types, dtype=torch.long)

    return out


class CEHRBERTModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        max_len=512,
        dropout=0.1,
        pooling="attention",      # "cls", "mean", "attention"
        use_token_type=True,
        n_event_types=4,          # 0 pad/unknown, 1 disease, 2 procedure, 3 medication
        name="CEHRBERT",
    ):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name = name
        self.pooling = pooling
        self.max_len = max_len
        self.use_token_type = use_token_type

        self.token_embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=0
        )

        self.position_embedding = nn.Embedding(max_len, embed_dim)

        if use_token_type:
            self.type_embedding = nn.Embedding(n_event_types, embed_dim)
        else:
            self.type_embedding = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        if pooling == "attention":
            self.attention = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, input_ids, attention_mask=None, token_types=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        positions = positions.expand(batch_size, seq_len)

        x = self.token_embedding(input_ids)
        x = x + self.position_embedding(positions)

        if self.use_token_type and token_types is not None:
            x = x + self.type_embedding(token_types)

        if attention_mask is None:
            attention_mask = (input_ids != 0).long()

        key_padding_mask = attention_mask == 0

        encoded = self.encoder(
            x,
            src_key_padding_mask=key_padding_mask
        )

        patient_embedding = self.pool(encoded, attention_mask)
        logits = self.classifier(self.dropout(patient_embedding)).view(-1)

        return logits, patient_embedding

    def pool(self, encoded, attention_mask):
        if self.pooling == "cls":
            return encoded[:, 0, :]

        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            summed = (encoded * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1e-8)
            return summed / denom

        if self.pooling == "attention":
            scores = self.attention(encoded).squeeze(-1)
            scores = scores.masked_fill(attention_mask == 0, -1e9)
            weights = torch.softmax(scores, dim=1)
            return torch.sum(encoded * weights.unsqueeze(-1), dim=1)

        raise ValueError(f"Unknown pooling: {self.pooling}")

    def train_model(
        self,
        train_loader,
        val_loader=None,
        num_epochs=10,
        lr=1e-4,
        weight_decay=1e-5,
        loss_type="focal",      # "focal" oppure "bce"
        alpha=0.85,
        gamma=2.0,
        enable_plot=False,
        frame_tqdm=None,
        frame_plot=None,
        plotsize=(6, 4),
    ):
        self.to(self.device)

        optimizer = torch.optim.Adam(self.parameters(),lr=lr,weight_decay=weight_decay)

        if loss_type == "focal":
            criterion = BinaryFocalLoss(alpha=alpha, gamma=gamma)
        else:
            criterion = nn.BCEWithLogitsLoss()

        train_losses, val_losses, val_aucs = [], [], []

        context = frame_tqdm if frame_tqdm is not None else nullcontext()

        with context:
            if frame_tqdm is not None:
                frame_tqdm.clear_output(wait=True)

            pbar = tqdm(
                range(num_epochs),
                total=num_epochs,
                desc=f"Training [{self.name}]"
            )

            for epoch in pbar:
                self.train()
                total_loss = 0.0

                for batch in train_loader:
                    optimizer.zero_grad()

                    input_ids = batch["input_ids"].long().to(self.device)
                    attention_mask = batch["attention_mask"].long().to(self.device)
                    y = batch["labels"].float().view(-1).to(self.device)

                    token_types = batch.get("token_types")
                    if token_types is not None:
                        token_types = token_types.long().to(self.device)

                    logits, _ = self(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_types=token_types
                    )

                    logits = logits.view(-1)

                    loss = criterion(logits, y)
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

                    optimizer.step()
                    total_loss += loss.item()

                avg_train_loss = total_loss / len(train_loader)
                train_losses.append(avg_train_loss)

                if val_loader is not None:
                    val_loss, auc = self.evaluate(val_loader, criterion=criterion)
                    val_losses.append(val_loss)
                    val_aucs.append(auc)

                    pbar.set_postfix({
                        "Train Loss": round(avg_train_loss, 3),
                        "Val Loss": round(val_loss, 3),
                        "AUC": round(auc, 3)
                    })
                else:
                    pbar.set_postfix({
                        "Train Loss": round(avg_train_loss, 3)
                    })

                if enable_plot and frame_plot is not None:
                    with frame_plot:
                        #frame_plot.clear_output(wait=True)
                        plot_foo(
                            self.name,
                            train_losses,
                            val_losses,
                            size=plotsize
                        )

        return train_losses, val_losses

                    
    def evaluate(self, loader, criterion=None):
        self.eval()

        if criterion is None:
            criterion = nn.BCEWithLogitsLoss()

        total_loss = 0.0
        y_true = []
        y_score = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].long().to(self.device)
                attention_mask = batch["attention_mask"].long().to(self.device)
                y = batch["labels"].float().view(-1).to(self.device)

                token_types = batch.get("token_types")
                if token_types is not None:
                    token_types = token_types.long().to(self.device)

                logits, _ = self(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_types=token_types
                )

                logits = logits.view(-1)

                loss = criterion(logits, y)
                total_loss += loss.item()

                probs = torch.sigmoid(logits)

                y_true.extend(y.detach().cpu().numpy())
                y_score.extend(probs.detach().cpu().numpy())

        avg_loss = total_loss / len(loader)

        try:
            auc = roc_auc_score(y_true, y_score)
        except ValueError:
            auc = 0.0

        return avg_loss, auc

    def get_embeddings(self, loader):
        self.to(self.device)
        self.eval()

        embeddings = []
        ids = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].long().to(self.device)
                attention_mask = batch["attention_mask"].long().to(self.device)

                token_types = batch.get("token_types")
                if token_types is not None:
                    token_types = token_types.long().to(self.device)

                _, patient_embedding = self(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_types=token_types
                )

                embeddings.append(patient_embedding.cpu())

                if "ids" in batch:
                    ids.extend(batch["ids"])

        embeddings = torch.cat(embeddings, dim=0).numpy()

        return embeddings, ids

    def predict_proba(self, loader):
        self.to(self.device)
        self.eval()

        ids = []
        probas = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].long().to(self.device)
                attention_mask = batch["attention_mask"].long().to(self.device)

                token_types = batch.get("token_types")
                if token_types is not None:
                    token_types = token_types.long().to(self.device)

                logits, _ = self(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_types=token_types
                )

                probs = torch.sigmoid(logits.view(-1))

                probas.extend(probs.cpu().numpy())

                if "ids" in batch:
                    ids.extend(batch["ids"])

        return np.array(probas), ids

    def predict(self, loader, threshold=0.5):
        probas, ids = self.predict_proba(loader)
        preds = (probas >= threshold).astype(int)
        return preds, ids  
#------------------------------------------------------------------------------------------------
# BEHRT
#------------------------------------------------------------------------------------------------

class BEHRTDataset(Dataset):
    def __init__(self, data_dict, labels_dict=None, code2id=None, max_len=128, max_age=119):
        """
        Args:
            data_dict (dict): {patient_id: [(list_eventi, data_str), ...]}
            labels_dict (dict): {patient_id: label} (opzionale)
            code2id (dict): mapping evento → int
            max_len (int): lunghezza massima della sequenza
            max_age (int): età massima per age embedding (troncata)
        """
        self.data_dict = data_dict
        self.labels_dict = labels_dict
        if code2id is None:
            all_events = set()
            for visits in data_dict.values():
                for events, _ in visits:
                    all_events.update(events)
            self.code2id = {code: idx+1 for idx, code in enumerate(sorted(all_events))}
            self.code2id["<PAD>"] = 0
        else:
            self.code2id = code2id
        self.max_len = max_len
        self.max_age = max_age
        
        self.patients = list(data_dict.keys())

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        pid = self.patients[idx]
        visits = self.data_dict[pid]

        # 🔸 Se il paziente non ha eventi → sequenza vuota con [PAD]
        if len(visits) == 0:
            input_ids = [0] * self.max_len
            age_ids = [0] * self.max_len
            segment_ids = [0] * self.max_len
            attention_mask = [0] * self.max_len
        else:
            token_seq = []
            age_seq = []
            
            # calcolo data iniziale per età relativa
            start_date = datetime.strptime(visits[0][1], "%Y-%m-%d")
            
            for events, date_str in visits:
                # tokenizzazione eventi
                token_ids = []
                for ev in events:
                    if ev in self.code2id:
                        token_ids.append(self.code2id[ev])
                if len(token_ids) == 0:
                    continue  # skip visita senza eventi validi

                # uso il primo evento della visita (BEHRT considera 1 evento per posizione)
                token_seq.append(token_ids[0])

                # calcolo età relativa
                event_date = datetime.strptime(date_str, "%Y-%m-%d")
                relative_years = (event_date - start_date).days // 365
                relative_years = max(0, min(self.max_age, relative_years))
                age_seq.append(relative_years)

            # padding/truncation
            if len(token_seq) > self.max_len:
                token_seq = token_seq[:self.max_len]
                age_seq = age_seq[:self.max_len]
            else:
                pad_len = self.max_len - len(token_seq)
                token_seq += [0] * pad_len
                age_seq += [0] * pad_len

            input_ids = token_seq
            age_ids = age_seq
            segment_ids = [0] * self.max_len  # no segment info
            attention_mask = [1 if t != 0 else 0 for t in input_ids]

        # conversione in tensori
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        age_ids = torch.tensor(age_ids, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        if self.labels_dict is not None:
            label = torch.tensor(self.labels_dict[pid], dtype=torch.long)
            return input_ids, age_ids, segment_ids, attention_mask, label
        else:
            return input_ids, age_ids, segment_ids, attention_mask


class BEHRTModel(nn.Module):
    def __init__(self, vocab_size, age_vocab_size=120, segment_vocab_size=2,
                 embed_dim=128, num_layers=2, num_heads=4, dropout=0.1, num_labels=2, name='BEHRT'):
        super().__init__()
        # 🔸 Embedding layers
        self.name = name
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.age_embedding = nn.Embedding(age_vocab_size, embed_dim, padding_idx=0)
        self.segment_embedding = nn.Embedding(segment_vocab_size, embed_dim, padding_idx=0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 🔸 Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 🔸 Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_labels)

    def forward(self, input_ids, age_ids, segment_ids, attention_mask):
        """
        input_ids: [B, L]
        age_ids: [B, L]
        segment_ids: [B, L]
        attention_mask: [B, L]
        """
        token_emb = self.token_embedding(input_ids)
        age_emb = self.age_embedding(age_ids)
        seg_emb = self.segment_embedding(segment_ids)

        x = token_emb + age_emb + seg_emb  # [B, L, H]

        # attention mask per Transformer: True=PAD → masked
        src_key_padding_mask = (attention_mask == 0)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        # usa il [CLS]-like → primo token
        pooled = x[:, 0, :]
        logits = self.classifier(self.dropout(pooled))

        return logits, pooled

    def train_model(self, 
                    train_loader, 
                    val_loader=None, 
                    num_epochs=5, 
                    lr=1e-4, 
                    enable_plot=False,
                    frame_tqdm=None,
                    frame_plot=None,plotsize=(4,3)
                    ):
        self.to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        train_losses = []
        val_losses = []
        self.train()
        train_loss = 0
        with frame_tqdm:
            frame_tqdm.clear_output(wait=True)
            pbar = tqdm(range(num_epochs), total=num_epochs, desc=f'Training [{self.name}]')
            for epoch in pbar:
                total_loss = 0
                for batch in train_loader:
                    input_ids, age_ids, segment_ids, attention_mask, labels = [b.to(self.device) for b in batch]

                    optimizer.zero_grad()
                    logits, _ = self(input_ids, age_ids, segment_ids, attention_mask)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    
                train_loss = total_loss / len(train_loader)
                train_losses.append(train_loss)

                # VALIDATION
                if val_loader is not None:
                    self.eval()
                    val_total_loss = 0

                    all_true = []
                    all_pred = []

                    with torch.no_grad():
                        for batch in val_loader:
                            input_ids, age_ids, segment_ids, attention_mask, labels = [
                                b.to(self.device) for b in batch
                            ]

                            logits, _ = self(input_ids, age_ids, segment_ids, attention_mask)
                            loss = criterion(logits, labels)

                            val_total_loss += loss.item()

                            probs = torch.softmax(logits, dim=1)[:, 1]
                            all_true.append(labels.cpu())
                            all_pred.append(probs.cpu())

                    val_loss = val_total_loss / len(val_loader)
                    val_losses.append(val_loss)

                    # AUROC
                    try:
                        y_true = torch.cat(all_true).numpy()
                        y_pred = torch.cat(all_pred).numpy()
                        auc = roc_auc_score(y_true, y_pred)
                    except:
                        auc = float("nan")
                        
                # PROGRESS BAR
                if val_loader:
                    pbar.set_postfix({'Train Loss': round(train_loss,3), 'Val Loss': round(val_loss,3), 'AUC': round(auc,3)})
                else:
                    pbar.set_postfix({'Train Loss': round(train_loss,3)})

                # OPTIONAL PLOT
                with frame_plot:
                    plot_foo(self.name, train_losses, val_losses, size=plotsize) if enable_plot else None

        return train_losses, val_losses

    @torch.no_grad()
    def predict_proba(self, dataloader):
        self.eval()
        self.to(self.device)

        all_probs = []

        for batch in dataloader:

            # 🔹 Gestione batch con o senza labels
            if len(batch) == 5:
                input_ids, age_ids, segment_ids, attention_mask, _ = batch
            elif len(batch) == 4:
                input_ids, age_ids, segment_ids, attention_mask = batch
            else:
                raise ValueError(f"Unexpected batch length: {len(batch)}")

            input_ids = input_ids.to(self.device)
            age_ids = age_ids.to(self.device)
            segment_ids = segment_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

            logits, _ = self(input_ids, age_ids, segment_ids, attention_mask)

            # 🔹 Binary case (num_labels = 1)
            if logits.shape[1] == 1:
                prob_pos = torch.sigmoid(logits)                  # (B,1)
                probs = torch.cat([1 - prob_pos, prob_pos], dim=1)  # (B,2)

            # 🔹 Multiclass (num_labels >= 2)
            else:
                probs = torch.softmax(logits, dim=1)              # (B,C)

            all_probs.append(probs.detach().cpu())

        return torch.cat(all_probs, dim=0)

    @torch.no_grad()
    def predict(self, dataloader):
        probs = self.predict_proba(dataloader)
        preds = torch.argmax(probs, dim=1)
        return preds, probs

    @torch.no_grad()
    def get_embeddings(self, dataloader):
        self.to(self.device)
        self.eval()
        all_ids = []
        all_embeds = []

        for batch in dataloader:
            if len(batch) == 5:
                input_ids, age_ids, segment_ids, attention_mask, labels = batch
            else:
                input_ids, age_ids, segment_ids, attention_mask = batch
                labels = None

            input_ids = input_ids.to(self.device)
            age_ids = age_ids.to(self.device)
            segment_ids = segment_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

            _, pooled = self(input_ids, age_ids, segment_ids, attention_mask)
            all_embeds.append(pooled.cpu().numpy())

        return np.concatenate(all_embeds, axis=0)
    
#------------------------------------------------------------------------------------------------
# GRU
#------------------------------------------------------------------------------------------------
class GRUModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, gru_hidden_dim=64, pooling="mean", name='GRU'):
        super().__init__()
        self.name = name
        self.pooling = pooling
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, gru_hidden_dim, batch_first=True)
        if pooling == "attention":
            self.attention = nn.Linear(gru_hidden_dim, 1)
        # aggiungo layer finale
        self.fc_out = nn.Linear(gru_hidden_dim, 1)
        self.to(self.device)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, hn = self.gru(embedded)
        if self.pooling == "mean":
            pooled = output.mean(dim=1)
        elif self.pooling == "last":
            pooled = hn.squeeze(0)
        elif self.pooling == "max":
            pooled = output.max(dim=1).values
        else:
            raise ValueError(f"Pooling {self.pooling} non supportato")
        logits = self.fc_out(pooled)  # proietto su 1 dimensione
        return logits
        
    def train_model(self, 
                    train_loader, 
                    val_loader=None, 
                    num_epochs=10, 
                    lr=0.01, 
                    enable_plot=False,
                    frame_tqdm=None,
                    frame_plot=None,plotsize=(4,3)):
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()
        train_losses, val_losses, val_aucs = [], [], []

        with frame_tqdm:
            frame_tqdm.clear_output(wait=True)
            pbar = tqdm(range(num_epochs), total=num_epochs, desc=f'Training [{self.name}]')
            for epoch in pbar:
                self.train()
                total_loss = 0

                for batch in train_loader:
                    optimizer.zero_grad()
                    # Batch può essere (id, x) o (id, x, y)
                    if len(batch) == 3:
                        ids, x, y = batch
                        y = y.float().unsqueeze(1).to(self.device)
                    elif len(batch) == 2:
                        ids, x = batch
                        y = None
                    else:
                        raise ValueError("Batch deve essere (id,x) o (id,x,y)")

                    x = x.long().to(self.device)
                    logits = self.forward(x)

                    if y is not None:
                        loss = criterion(logits, y)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()

                train_loss = total_loss / len(train_loader) if total_loss > 0 else 0
                train_losses.append(train_loss)

                # VALIDATION
                if val_loader is not None:
                    val_loss, val_auc = self.evaluate(val_loader)
                    val_losses.append(val_loss)
                    val_aucs.append(val_auc)
                    pbar.set_postfix({'Train Loss': round(train_loss,3), 'Val Loss': round(val_loss,3), 'AUC': round(val_auc,3)})
                else:
                    pbar.set_postfix({'Train Loss': round(train_loss,3)})

                with frame_plot:
                    plot_foo(self.name, train_losses, val_losses, size=plotsize) if enable_plot else None

        return train_losses, val_losses, val_aucs

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.eval()
        criterion = nn.BCEWithLogitsLoss()
        total_loss = 0
        preds, trues = [], []

        for batch in dataloader:
            if len(batch) == 3:
                ids, x, y = batch
                y = y.float().unsqueeze(1).to(self.device)
            elif len(batch) == 2:
                ids, x = batch
                y = None
            else:
                raise ValueError("Batch deve essere (id,x) o (id,x,y)")

            x = x.long().to(self.device)
            logits = self.forward(x)

            if y is not None:
                loss = criterion(logits, y)
                total_loss += loss.item()
                preds.extend(torch.sigmoid(logits).cpu().numpy())
                trues.extend(y.cpu().numpy())

        avg_loss = total_loss / len(dataloader) if total_loss > 0 else 0
        try:
            auc = roc_auc_score(trues, preds) if len(trues) > 0 else float("nan")
        except:
            auc = float("nan")

        return avg_loss, auc

    @torch.no_grad()
    def predict_proba(self, dataloader):
        self.eval()

        all_probs = []

        for batch in dataloader:
            if len(batch) == 3:
                ids, x, _ = batch
            elif len(batch) == 2:
                ids, x = batch
            else:
                raise ValueError("Batch deve essere (id,x) o (id,x,y)")

            x = x.long().to(self.device)
            logits = self.forward(x)

            # 🔹 Binary case (BCEWithLogitsLoss)
            if logits.shape[1] == 1:
                prob_pos = torch.sigmoid(logits)                  # (batch, 1)
                probs = torch.cat([1 - prob_pos, prob_pos], dim=1)  # (batch, 2)

            # 🔹 Multiclass case (se mai lo userai)
            else:
                probs = torch.softmax(logits, dim=1)              # (batch, C)

            all_probs.append(probs.cpu())

        return torch.cat(all_probs, dim=0)

    @torch.no_grad()
    def predict(self, dataloader):
        probs = self.predict_proba(dataloader)
        preds = torch.argmax(probs, dim=1)
        return preds, probs  
    
    @torch.no_grad()
    def get_embeddings(self, dataloader):
        """
        Restituisce embeddings e ids come array numpy
        """
        self.eval()
        all_feats, all_ids = [], []

        for batch in dataloader:
            if len(batch) >= 2:
                ids, x = batch[:2]
            else:
                raise ValueError("Batch deve avere almeno (id,x)")

            x = x.long().to(self.device)
            feats = self.forward(x)
            all_feats.append(feats.cpu().numpy())
            all_ids.extend(ids)

        embeddings = np.vstack(all_feats)
        ids_array = np.array(all_ids)
        return embeddings, ids_array

#------------------------------------------------------------------------------------------------
# GRU-D
#------------------------------------------------------------------------------------------------
def grud_collate_fn(batch):
    """
    Collate function per GRU-D con padding dinamico.
    Funziona sia per dataset con labels che senza.
    """
    # controlla se il primo elemento ha labels (5 elementi totali)
    has_labels = len(batch[0]) == 5

    if has_labels:
        xs, ms, deltas, labels, ids = zip(*batch)
    else:
        xs, ms, deltas, ids = zip(*batch)
        labels = None

    xs_padded = pad_sequence(xs, batch_first=True, padding_value=0.0)
    ms_padded = pad_sequence(ms, batch_first=True, padding_value=0.0)
    deltas_padded = pad_sequence(deltas, batch_first=True, padding_value=0.0)

    if labels is not None:
        labels = torch.tensor(labels, dtype=torch.long)
        return xs_padded, ms_padded, deltas_padded, labels, ids
    else:
        return xs_padded, ms_padded, deltas_padded, ids
    
class GRUDDataset(Dataset):
    def __init__(self, patient_dict, code2id=None, labels_dict=None, max_seq_len=50):
        """
        patient_dict: {pid: [( [event1, event2, ...], 'YYYY-MM-DD'), ...]}
        code2id: dict evento → indice intero
        labels_dict: opzionale {pid: 0/1/...}
        """
        self.patient_ids = list(patient_dict.keys())
        self.data = patient_dict
        self.labels = labels_dict
        if code2id is None:
            all_events = set()
            for visits in patient_dict.values():
                for events, _ in visits:
                    all_events.update(events)
            self.code2id = {code: idx+1 for idx, code in enumerate(sorted(all_events))}
            self.code2id["<PAD>"] = 0
        else:
            self.code2id = code2id
        self.max_seq_len = max_seq_len
        self.input_size = len(code2id)

        # Calcolo media feature per imputazione GRU-D
        self.feature_means = np.zeros(self.input_size, dtype=np.float32)
        counts = np.zeros(self.input_size, dtype=np.float32)
        for visits in patient_dict.values():
            for events, _ in visits:
                for ev in events:
                    if ev in code2id:
                        idx = code2id[ev]
                        self.feature_means[idx] += 1
                        counts[idx] += 1
        counts[counts == 0] = 1
        self.feature_means /= counts

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        visits = self.data[pid]

        # Ordina per data
        visits = sorted(visits, key=lambda x: x[1])

        x = np.zeros((self.max_seq_len, self.input_size), dtype=np.float32)
        m = np.zeros_like(x)
        delta = np.zeros_like(x)
        timestamps = []

        for t, (events, date_str) in enumerate(visits[:self.max_seq_len]):
            for ev in events:
                if ev in self.code2id:
                    eid = self.code2id[ev]
                    x[t, eid] = 1.0
                    m[t, eid] = 1.0
            try:
                timestamps.append(datetime.strptime(date_str, "%Y-%m-%d"))
            except:
                timestamps.append(datetime.now())  # fallback se data mancante

        # Calcolo delta temporali
        if len(timestamps) > 0:
            for d in range(self.input_size):
                last_time = timestamps[0]
                for t in range(1, len(timestamps)):
                    if m[t-1, d] == 1:
                        last_time = timestamps[t-1]
                    delta[t, d] = (timestamps[t] - last_time).days

        x_tensor = torch.tensor(x, dtype=torch.float32)
        m_tensor = torch.tensor(m, dtype=torch.float32)
        delta_tensor = torch.tensor(delta, dtype=torch.float32)

        if self.labels is not None and pid in self.labels:
            y = torch.tensor(self.labels[pid], dtype=torch.long)
            return x_tensor, m_tensor, delta_tensor, y, pid
        else:
            return x_tensor, m_tensor, delta_tensor, pid

class GRUDModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=2, x_mean=None, dropout=0.5, name='GRU-D'):
        super().__init__()
        self.name = name
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.x_mean = torch.tensor(x_mean, dtype=torch.float32) if x_mean is not None else torch.zeros(input_size)

        # GRU-D
        self.gru_cell = nn.GRUCell(input_size * 2, hidden_size)  # x_hat concatenato con mask
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, m, delta):
        batch_size, seq_len, input_size = x.size()
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)

        # imputazione x_hat con feature mean
        x_hat = m * x + (1 - m) * self.x_mean.to(x.device)

        for t in range(seq_len):
            x_t_hat = x_hat[:, t, :]
            m_t = m[:, t, :]
            gru_input = torch.cat([x_t_hat, m_t], dim=1)
            h = self.gru_cell(gru_input, h)

        h = self.dropout(h)
        out = self.fc(h)
        return out

    def train_model(self, 
                    dataloader, 
                    val_loader=None, 
                    num_epochs=10, 
                    lr=1e-3, enable_plot=False,
                    frame_tqdm=None,
                    frame_plot=None,plotsize=(4,3)):
        
        self.to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        train_losses = []
        val_losses = []
        train_loss = 0
        with frame_tqdm:
            frame_tqdm.clear_output(wait=True)
            pbar = tqdm(range(num_epochs), total=num_epochs, desc=f'Training [{self.name}]')
            for epoch in pbar:
                self.train()
                epoch_loss = 0
                #pbar = tqdm(dataloader, desc=f"[GRU-D] Epoch {epoch+1}/{num_epochs} Loss {avg_loss:.4f}")
                for batch in dataloader:
                    if len(batch) == 5:
                        x, m, delta, y, _ = batch
                        y = y.to(self.device)
                    else:
                        continue  # skip test batches

                    x, m, delta = x.to(self.device).float(), m.to(self.device).float(), delta.to(self.device).float()

                    optimizer.zero_grad()
                    logits = self(x, m, delta)
                    loss = criterion(logits, y)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                train_loss = epoch_loss / len(dataloader)
                train_losses.append(train_loss)
                
                # VALIDATION
                if val_loader is not None:
                    self.eval()
                    val_total_loss = 0
                    all_true = []
                    all_pred = []

                    with torch.no_grad():
                        for batch in val_loader:
                            # skip test batches
                            if len(batch) != 5:
                                continue

                            x, m, delta, y, _ = batch
                            x = x.to(self.device).float()
                            m = m.to(self.device).float()
                            delta = delta.to(self.device).float()
                            y = y.to(self.device)

                            logits = self(x, m, delta)
                            loss = criterion(logits, y)
                            val_total_loss += loss.item()

                            probs = torch.softmax(logits, dim=1)[:, 1]
                            all_true.append(y.cpu())
                            all_pred.append(probs.cpu())

                    val_loss = val_total_loss / len(val_loader)
                    val_losses.append(val_loss)

                    # compute AUROC if possible
                    try:
                        y_true = torch.cat(all_true).numpy()
                        y_pred = torch.cat(all_pred).numpy()
                        auc = roc_auc_score(y_true, y_pred)
                    except:
                        auc = float("nan")

                # PROGRESS BAR
                if val_loader:
                    pbar.set_postfix({'Train Loss': round(train_loss,3), 'Val Loss': round(val_loss,3), 'AUC': round(auc,3)})
                else:
                    pbar.set_postfix({'Train Loss': round(train_loss,3)})

                # PLOT (opzional)
                with frame_plot:
                    plot_foo(self.name, train_losses, val_losses, size=plotsize) if enable_plot else None

        return train_losses, val_losses

    @torch.no_grad()
    def predict_proba(self, dataloader):
        self.eval()
        all_probs = []

        for batch in dataloader:
            # 🔹 Gestione batch coerente con train_model
            if len(batch) == 5:
                x, m, delta, _, _ = batch
            elif len(batch) == 4:
                x, m, delta, _ = batch
            else:
                raise ValueError(f"Unexpected batch length: {len(batch)}")

            x = x.to(self.device).float()
            m = m.to(self.device).float()
            delta = delta.to(self.device).float()

            logits = self.forward(x, m, delta)

            # 🔹 Binary (output_size = 1)
            if logits.shape[1] == 1:
                prob_pos = torch.sigmoid(logits)
                probs = torch.cat([1 - prob_pos, prob_pos], dim=1)
            # 🔹 Multiclass (output_size >= 2)
            else:
                probs = torch.softmax(logits, dim=1)

            all_probs.append(probs.detach().cpu())

        return torch.cat(all_probs, dim=0)

    @torch.no_grad()
    def predict(self, dataloader):
        probs = self.predict_proba(dataloader)
        preds = torch.argmax(probs, dim=1)
        return preds, probs

    def get_embeddings(self, dataloader):
        self.to(self.device)
        self.eval()
        all_embeds = []
        all_ids = []

        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 5:
                    x, m, delta, _, pid = batch
                else:
                    x, m, delta, pid = batch

                x, m, delta = x.to(self.device).float(), m.to(self.device).float(), delta.to(self.device).float()
                batch_size, seq_len, input_size = x.size()
                h = torch.zeros(batch_size, self.hidden_size, device=self.device)
                x_hat = m * x + (1 - m) * self.x_mean.to(x.device)
                for t in range(seq_len):
                    x_t_hat = x_hat[:, t, :]
                    m_t = m[:, t, :]
                    gru_input = torch.cat([x_t_hat, m_t], dim=1)
                    h = self.gru_cell(gru_input, h)
                all_embeds.append(h.cpu())
                all_ids.extend(pid)
        return torch.cat(all_embeds, dim=0).numpy(), all_ids

#------------------------------------------------------------------------------------------------
# Dipole
#------------------------------------------------------------------------------------------------

class DipoleDataset(Dataset):
    """
    Input format:
        series = {
            patient_id: [[event_codes], [event_codes], ...],   # each inner list = visit
        }
        labels = { patient_id: 0/1 } OR None
    """

    def __init__(self, series, labels=None, code2id=None, ignore_token='[PAD]'):
        self.ids = list(series.keys())
        self.series = series
        self.labels = labels
        self.code2id = code2id

        # Build vocabulary if needed
        if code2id is None:
            self.code2id = {ignore_token: 0}
            idx = 1
            for pid in self.ids:
                for visit in series[pid]:
                    for code in visit:
                        if code not in self.code2id:
                            self.code2id[code] = idx
                            idx += 1

        self.pad_id = self.code2id[ignore_token]

    def __len__(self):
        return len(self.ids)

    def encode_visit(self, visit):
        return [self.code2id.get(c, self.pad_id) for c in visit]

    def __getitem__(self, idx):
        patient_id = self.ids[idx]
        visits = self.series[patient_id]
        encoded = [self.encode_visit(v) for v in visits]

        if self.labels is None:
            return encoded, None, patient_id
        else:
            return encoded, self.labels[patient_id], patient_id
        
def dipole_collate(batch, pad_id=0):
    """
    batch: list of tuples (encoded_visits, label or None, patient_id)
    """

    sequences = [b[0] for b in batch]
    labels = [b[1] for b in batch]
    ids = [b[2] for b in batch]

    # 1) max_visits
    max_visits = max(len(seq) for seq in sequences)

    # 2) max_codes per ogni visit
    max_codes = 0
    for seq in sequences:
        for v in seq:
            max_codes = max(max_codes, len(v))

    padded_X = []
    padded_M = []

    for seq in sequences:
        # mask: visit-level mask (1 if visit exists)
        visit_mask = [1] * len(seq)

        # pad visits
        while len(seq) < max_visits:
            seq.append([pad_id])
            visit_mask.append(0)

        # pad codes
        padded_visits = []
        padded_code_masks = []
        for visit in seq:
            code_mask = [1] * len(visit)
            while len(visit) < max_codes:
                visit.append(pad_id)
                code_mask.append(0)
            padded_visits.append(visit)
            padded_code_masks.append(code_mask)

        padded_X.append(padded_visits)
        padded_M.append([visit_mask, padded_code_masks])

    X = torch.tensor(padded_X, dtype=torch.long)
    M_visit = torch.tensor([m[0] for m in padded_M], dtype=torch.float)
    M_code = torch.tensor([m[1] for m in padded_M], dtype=torch.float)

    # return with or without labels
    if all(l is None for l in labels):
        return X, (M_visit, M_code), None, ids
    else:
        y = torch.tensor(labels, dtype=torch.float)
        return X, (M_visit, M_code), y, ids

class DipoleAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, H, M):
        """
        H: (B, T, H)
        M: (M_visit, M_code)
            M_visit: (B, T)  visit-level mask
        """
        M_visit, _ = M

        # u_t = tanh(W h_t)
        score = torch.tanh(self.W(H))   # (B, T, H)

        # raw attention scores
        score = self.v(score).squeeze(-1)  # (B, T)

        # Apply visit mask
        score = score.masked_fill(M_visit == 0, -1e9)

        # Attention weights
        attn = torch.softmax(score, dim=1)   # (B, T)

        # Context (B, H)
        context = torch.sum(attn.unsqueeze(-1) * H, dim=1)

        return context, attn


class DipoleModel(nn.Module):

    def __init__(self, vocab_size, embed_size=128, hidden_size=128,
                 num_layers=1, output_size=2, dropout=0.3, pad_idx=0, name='Dipole'):
        super().__init__()
        self.name = name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.rnn = nn.GRU(
            embed_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.attention = DipoleAttention(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, X, M):
        """
        X: (B, T, Vmax)
        M: (B, T)
        """
        B, T, Vmax = X.shape

        # embed visits by mean-pooling event codes
        emb = self.embedding(X)                        # (B, T, Vmax, E)
        emb = emb.mean(dim=2)                          # (B, T, E)

        # RNN
        H, _ = self.rnn(emb)                           # (B, T, H*2)

        # Attention
        context, attn = self.attention(H, M)

        out = self.dropout(context)
        logits = self.fc(out)

        return logits, context, attn

    # ------------------------------------------------------------
    #               TRAINING FUNCTION with PLOT
    # ------------------------------------------------------------

    def train_model(self, train_loader, val_loader=None,
                    lr=1e-3, num_epochs=10,
                    enable_plot=True, early_stopping=False,
                    patience=5,
                    frame_tqdm=None,
                    frame_plot=None,plotsize=(4,3)):

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.to(self.device)
        train_losses = []
        val_losses = []

        best_val = float("inf")
        patience_ctr = 0

        with frame_tqdm:
            frame_tqdm.clear_output(wait=True)
            pbar = tqdm(range(num_epochs), total=num_epochs, desc=f'Training [{self.name}]')
            for epoch in pbar:
                # TRAIN
                self.train()
                total_loss = 0

                for batch in train_loader:
                    # Gestione batch con o senza M / y
                    if len(batch) == 4:
                        X, M, y, _ = batch
                    elif len(batch) == 3:
                        X, M, y = batch
                    else:
                        raise ValueError(f"Unexpected batch length: {len(batch)}")

                    X = X.to(self.device)

                    if M is not None:
                        M = (M[0].to(self.device), M[1].to(self.device))

                    if y is not None:
                        y = y.long().to(self.device)   # <-- conversione a LongTensor

                    optimizer.zero_grad()
                    logits, _, _ = self(X, M)

                    loss = criterion(logits, y)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                train_loss = total_loss / len(train_loader)
                #pbar.set_description(f"[Dipole] Embedding: Loss {avg_loss:.4f}")
                train_losses.append(train_loss)

                # VALIDATION
                if val_loader is not None:
                    self.eval()
                    val_total_loss = 0

                    with torch.no_grad():
                        for batch in val_loader:
                            if len(batch) == 4:
                                X, M, y, _ = batch
                            elif len(batch) == 3:
                                X, M, y = batch
                            else:
                                raise ValueError(f"Unexpected batch length: {len(batch)}")

                            X = X.to(self.device)
                            if M is not None:
                                M = (M[0].to(self.device), M[1].to(self.device))
                            y = y.long().to(self.device)

                            logits, _, _ = self(X, M)
                            loss = criterion(logits, y)
                            val_total_loss += loss.item()

                    val_loss = val_total_loss / len(val_loader)
                    val_losses.append(val_loss)

                    # early stopping
                    if val_loss < best_val:
                        best_val = val_loss
                        patience_ctr = 0
                        best_state = {
                            "model": self.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "epoch": epoch
                        }
                    else:
                        patience_ctr += 1

                    if early_stopping and patience_ctr >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        self.load_state_dict(best_state["model"])
                        break
                    
                # PROGRESS BAR
                if val_loader:
                    pbar.set_postfix({'Train Loss': round(train_loss,3), 'Val Loss': round(val_loss,3), 'AUC': round(auc,3)})
                else:
                    pbar.set_postfix({'Train Loss': round(train_loss,3)})

                # PLOT (opzional)
                with frame_plot:
                    plot_foo(self.name, train_losses, val_losses, size=plotsize) if enable_plot else None

        return train_losses, val_losses

    # ------------------------------------------------------------
    #               PREDICTION FUNCTION
    # ------------------------------------------------------------

    @torch.no_grad()
    def predict_proba(self, dataloader):
        self.eval()
        self.to(self.device)

        all_probs = []

        for batch in dataloader:

            # 🔹 ALLINEATO A train_model
            if len(batch) == 4:
                X, M, y, ids = batch
            elif len(batch) == 3:
                X, M, ids = batch
            else:
                raise ValueError(f"Unexpected batch length: {len(batch)}")

            # 🔹 X è sempre tensor
            X = X.long().to(self.device)

            # 🔹 FIX CRUCIALE: M può essere tuple
            if M is not None:
                if isinstance(M, tuple):
                    M = (M[0].to(self.device), M[1].to(self.device))
                else:
                    M = M.to(self.device)

            logits, _, _ = self(X, M)

            # 🔹 Binary
            if logits.shape[1] == 1:
                prob_pos = torch.sigmoid(logits)
                probs = torch.cat([1 - prob_pos, prob_pos], dim=1)

            # 🔹 Multiclass
            else:
                probs = torch.softmax(logits, dim=1)

            all_probs.append(probs.detach().cpu())

        return torch.cat(all_probs, dim=0)

    @torch.no_grad()
    def predict(self, dataloader):
        probs = self.predict_proba(dataloader)
        preds = torch.argmax(probs, dim=1)
        return preds, probs

    # ------------------------------------------------------------
    #         EXTRACT EMBEDDINGS (context vectors)
    # ------------------------------------------------------------

    @torch.no_grad()
    def get_embeddings(self, loader):
        self.eval()
        all_embeds = []
        all_ids = []

        with torch.no_grad():
            for batch in loader:
                # batch può essere con o senza label
                if len(batch) == 4:
                    X, M, _, ids = batch
                elif len(batch) == 3:
                    X, M, ids = batch
                else:
                    raise ValueError(f"Unexpected batch length: {len(batch)}")

                X = X.to(self.device)
                if M is not None:
                    M = (M[0].to(self.device), M[1].to(self.device))

                _, context, _ = self(X, M)  # context = embedding
                all_embeds.append(context.cpu())
                all_ids.extend(ids)

        all_embeds = torch.cat(all_embeds, dim=0)
        return all_embeds.numpy(), all_ids


# --------------------------------------------
# TLSTM
# --------------------------------------------

def timeaware_collate_fn(batch):
    """
    batch: lista di (pid, X_seq, T_seq, y)
    """
    ids = []
    ys = []
    X_batch = []
    M_batch = []
    delta_batch = []

    # Filtra batch malformati
    cleaned_batch = []
    for sample in batch:
        pid, X_seq, T_seq, y = sample

        if len(X_seq) == 0:
            #print(f"[WARNING] Paziente {pid} ha sequenza vuota. Inserisco timestep fittizio.")
            X_seq = [[0]]
            T_seq = [0]

        cleaned_batch.append((pid, X_seq, T_seq, y))

    batch = cleaned_batch
    lengths = [len(b[1]) for b in batch]
    max_len = max(lengths)

    for pid, X_seq, T_seq, y in batch:
        ids.append(pid)
        ys.append(y)

        padded_X = []
        padded_M = []
        padded_deltas = []

        # Pooling media degli eventi per timestep
        for x_step, dt in zip(X_seq, T_seq):

            if len(x_step) == 0:
                pooled = 0
            else:
                pooled = sum(x_step) / len(x_step)

            padded_X.append(pooled)
            padded_M.append(1)
            padded_deltas.append(dt)

        # Padding fino a max_len
        pad_len = max_len - len(X_seq)
        if pad_len > 0:
            padded_X.extend([0] * pad_len)
            padded_M.extend([0] * pad_len)
            padded_deltas.extend([0] * pad_len)

        X_batch.append(padded_X)
        M_batch.append(padded_M)
        delta_batch.append(padded_deltas)

    X_batch = torch.tensor(X_batch, dtype=torch.long)
    M_batch = torch.tensor(M_batch, dtype=torch.float)
    delta_batch = torch.tensor(delta_batch, dtype=torch.float)
    ys = torch.tensor(ys, dtype=torch.float)

    return X_batch, M_batch, delta_batch, ys, ids


# =========================
# DATASET
# =========================
class Med2VecDataset(Dataset):
    def __init__(self, sequences_dict, labels_dict=None, word_to_idx=None):
        self.ids = list(sequences_dict.keys())

        # ✅ encoding UNA SOLA VOLTA (non in __getitem__)
        self.sequences = []
        for _id in self.ids:
            visits = sequences_dict[_id]

            encoded_visits = []
            for visit in visits:
                encoded_visits.append([word_to_idx.get(c, 0) for c in visit])

            self.sequences.append(encoded_visits)

        if labels_dict is not None:
            self.labels = [torch.tensor(labels_dict[_id], dtype=torch.float) for _id in self.ids]
        else:
            self.labels = None

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        _id = self.ids[idx]
        visits = self.sequences[idx]

        # ✅ gestione sequenze vuote
        if len(visits) == 0:
            visits = [[0]]

        if self.labels is None:
            return _id, visits
        else:
            return _id, visits, self.labels[idx]


# =========================
# COLLATE FUNCTION
# =========================
def med2vec_collate(batch, pad_token=0):
    ids = []
    sequences = []
    labels = []

    has_labels = len(batch[0]) == 3

    for item in batch:
        if has_labels:
            _id, visits, y = item
            labels.append(y)
        else:
            _id, visits = item

        ids.append(_id)
        sequences.append(visits)

    max_visits = max(len(seq) for seq in sequences)
    max_codes = max(len(v) for seq in sequences for v in seq)

    padded = []
    for seq in sequences:
        padded_seq = []

        for visit in seq:
            visit = list(visit)
            v = visit + [pad_token] * (max_codes - len(visit))
            padded_seq.append(v)

        for _ in range(max_visits - len(seq)):
            padded_seq.append([pad_token] * max_codes)

        padded.append(padded_seq)

    x = torch.tensor(padded, dtype=torch.long)  # (B, V, C)

    if has_labels:
        y = torch.stack(labels)
        return ids, x, y
    else:
        return ids, x


# =========================
# MODEL
# =========================
class Med2VecModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, visit_dim, name='Med2Vec'):
        super().__init__()
        self.name = name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Code embedding
        self.code_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Visit embedding (MLP)
        self.visit_layer = nn.Linear(embed_dim, visit_dim)

        # Visit prediction (skip-gram style)
        self.output_layer = nn.Linear(visit_dim, vocab_size)

        # Classifier
        self.classifier = nn.Linear(visit_dim, 1)

    def forward(self, x):
        # x: (B, V, C)
        emb = self.code_embedding(x)  # (B, V, C, D)

        mask = (x != 0).unsqueeze(-1)
        emb = emb * mask

        # sum codes → visit embedding
        visit_emb = emb.sum(dim=2)  # (B, V, D)

        # MLP + ReLU
        visit_emb = torch.relu(self.visit_layer(visit_emb))  # (B, V, visit_dim)

        # patient embedding
        patient_emb = visit_emb.mean(dim=1)  # (B, visit_dim)

        logits = self.classifier(patient_emb)

        return logits, patient_emb, visit_emb


    def compute_visit_loss(self, visit_emb, x):
        B, V, _ = visit_emb.shape
        losses = []

        for t in range(V - 1):
            vt = visit_emb[:, t, :]
            target_codes = x[:, t+1, :]  # (B, C)
            target = torch.zeros(x.size(0), self.output_layer.out_features, device=x.device)
            target.scatter_(1, target_codes, 1.0)
            target[:, 0] = 0  # ignora padding
            logits = self.output_layer(vt)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            losses.append(loss)

        if len(losses) == 0:
            return torch.tensor(0.0, device=self.device)

        return torch.stack(losses).mean()


    # =========================
    # TRAIN
    # =========================
    def train_model(self, 
                    train_loader, 
                    val_loader=None, 
                    num_epochs=10, 
                    lr=1e-3,
                    lambda_visit=0.1,
                    enable_plot=False,
                    frame_tqdm=None,
                    frame_plot=None,
                    plotsize=(4,5)):

        self.to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        train_losses, val_losses, val_aucs = [], [], []

        with frame_tqdm:
            frame_tqdm.clear_output(wait=True)

            pbar = tqdm(range(num_epochs), total=num_epochs, desc=f"Training [{self.name}]")

            for epoch in pbar:
                # =========================
                # TRAIN
                # =========================
                self.train()
                total_loss = 0

                for batch in train_loader:
                    optimizer.zero_grad()

                    ids, x, y = batch

                    x = x.long().to(self.device)
                    y = y.float().unsqueeze(1).to(self.device)

                    logits, patient_emb, visit_emb = self(x)

                    loss_cls = criterion(logits, y)
                    loss_visit = self.compute_visit_loss(visit_emb, x)

                    loss = loss_cls + lambda_visit * loss_visit

                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                avg_train_loss = total_loss / len(train_loader)
                train_losses.append(avg_train_loss)

                # =========================
                # VALIDATION
                # =========================
                if val_loader is not None:
                    val_loss, auc = self.evaluate(val_loader)
                    val_losses.append(val_loss)
                    val_aucs.append(auc)

                    pbar.set_postfix({
                        'Train Loss': round(avg_train_loss, 3),
                        'Val Loss': round(val_loss, 3),
                        'AUC': round(auc, 3)
                    })
                else:
                    pbar.set_postfix({'Train Loss': round(avg_train_loss, 3)})

                # =========================
                # PLOT LIVE
                # =========================
                with frame_plot:
                    if enable_plot:
                        plot_foo(self.name, train_losses, val_losses, size=plotsize)

        return train_losses, val_losses

    # =========================
    # EVALUATE
    # =========================
    @torch.no_grad()
    def predict_proba(self, dataloader):
        """
        Ritorna le probabilità (sigmoid) per ogni paziente.
        
        Output:
            probs: np.array shape (N,)
        """
        self.eval()
        all_probs = []

        for batch in dataloader:
            if len(batch) == 3:
                ids, x, y = batch
            else:
                ids, x = batch

            x = x.long().to(self.device)

            logits, _, _ = self(x)  # (B, 1)
            # 🔹 Binary case (BCEWithLogitsLoss)
            if logits.shape[1] == 1:
                prob_pos = torch.sigmoid(logits)                  # (batch, 1)
                probs = torch.cat([1 - prob_pos, prob_pos], dim=1)  # (batch, 2)

            # 🔹 Multiclass case (se mai lo userai)
            else:
                probs = torch.softmax(logits, dim=1)              # (batch, C)

            all_probs.append(probs.cpu())

        return torch.cat(all_probs, dim=0)

    @torch.no_grad()
    def predict(self, dataloader, threshold=0.5):
        """
        Predizione con soglia custom.

        Args:
            threshold (float): soglia tra 0 e 1
            return_probs (bool): se True ritorna anche le probabilità

        Returns:
            preds oppure (preds, probs)
        """

        probs = self.predict_proba(dataloader)  # (N,2) oppure (N,C)
        # ----------------------------
        # BINARIO
        # ----------------------------
        if probs.shape[1] == 2:
            prob_pos = probs[:, 1]

            if not (0.0 <= threshold <= 1.0):
                raise ValueError("threshold must be in [0,1]")

            preds = (prob_pos >= threshold).long()

        # ----------------------------
        # MULTICLASS (fallback)
        # ----------------------------
        else:
            preds = torch.argmax(probs, dim=1)

        return preds, probs
    
    # =========================
    # EVALUATE
    # =========================
    @torch.no_grad()
    def evaluate(self, dataloader):
        self.eval()
        criterion = nn.BCEWithLogitsLoss()

        total_loss = 0
        preds, trues = [], []

        for batch in dataloader:
            ids, x, y = batch

            x = x.long().to(self.device)
            y = y.float().unsqueeze(1).to(self.device)

            logits, _, _ = self(x)

            loss = criterion(logits, y)
            total_loss += loss.item()

            probs = torch.sigmoid(logits).cpu().numpy()
            preds.extend(probs)
            trues.extend(y.cpu().numpy())

        avg_loss = total_loss / len(dataloader)

        try:
            auc = roc_auc_score(trues, preds)
        except:
            auc = float("nan")

        return avg_loss, auc


    # =========================
    # EMBEDDINGS
    # =========================
    @torch.no_grad()
    def get_embeddings(self, dataloader):
        self.eval()
        all_feats = []
        all_ids = []

        for batch in dataloader:
            ids, x = batch[:2]

            x = x.long().to(self.device)
            _, feats, _ = self(x)

            all_feats.append(feats.cpu().numpy())
            all_ids.extend(ids)

        return np.vstack(all_feats), np.array(all_ids)
