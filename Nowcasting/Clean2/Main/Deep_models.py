import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import copy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🖥️ Using device:", device)



class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=6, dropout=0, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=dropout, num_layers=num_layers)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :])
    

def create_sequences(X, y, lookback=6):
    X_seq, y_seq = [], []
    for i in range(lookback, len(X)):
        X_seq.append(X[i - lookback:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)


def train_lstm_model(X_train, y_train, X_test, lookback=6, hidden_size=6, dropout=0, epochs=15, batch_size=16, patience=3, num_layers=1):

    # 1. Concaténation train + test
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, [np.nan]*len(X_test)])  # juste pour être sur de pas montrer y_test

    # 2. Création des séquences sur tout
    X_seq_all, y_seq_all = create_sequences(X_all, y_all, lookback)

    # 3. Séquence test = la dernière
    X_test_seq = X_seq_all[-1:]

    # 4. Entraînement = toutes les autres
    X_seq_trainval = X_seq_all[:-1]
    y_seq_trainval = y_seq_all[:-1]

    # 5. Split train / val (10% dernières)
    val_size = max(10, int(0.1 * len(X_seq_trainval)))
    X_train_seq = X_seq_trainval[:-val_size]
    y_train_seq = y_seq_trainval[:-val_size]
    X_val_seq   = X_seq_trainval[-val_size:]
    y_val_seq   = y_seq_trainval[-val_size:]

    

    # 6. Passage en tenseurs pour torch
    X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32).view(-1, 1).to(device)
    X_val_tensor   = torch.tensor(X_val_seq, dtype=torch.float32).to(device)
    y_val_tensor   = torch.tensor(y_val_seq, dtype=torch.float32).view(-1, 1).to(device)
    X_test_tensor  = torch.tensor(X_test_seq, dtype=torch.float32).to(device)


    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size)

    model = SimpleLSTM(input_size=X_train.shape[1], hidden_size=hidden_size, dropout=dropout, num_layers=num_layers).to(device)

    # Gestion du déséquilibre
    n_pos = np.sum(y_train)
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model = None
    counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = loss_fn(preds, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_losses.append(total_loss / len(train_loader))


        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                preds = model(X_batch)
                loss = loss_fn(preds, y_batch)
                val_loss += loss.item()
                
        val_losses.append(val_loss / len(val_loader))

        # --- Early stoping
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            counter = 0
            best_model = copy.deepcopy(model.state_dict())
        else:
            counter += 1
            if counter >= patience:
                #print("⏹️ Early stopping triggered.")
                break

    if best_model is not None:
        model.load_state_dict(best_model)
    '''else:
        print("⚠️ pas de best_model")'''


    # Prédiction
    model.eval()
    with torch.no_grad():
        logit = model(X_test_tensor)
        proba = torch.sigmoid(logit).item()

    return model, proba, train_losses, val_losses
