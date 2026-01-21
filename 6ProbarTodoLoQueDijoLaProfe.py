# Comparacion completa de tods los modelos con optimizacion de hiperparametros

# Ref:
#Optuna: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html
#LSTM: https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html
#Cnn:https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html (aunque en la web se llama cnv1d o algo asi)
#tranformer: https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import optuna
import time
import warnings
warnings.filterwarnings('ignore')
import random

# Referencia: https://pytorch.org/docs/stable/notes/randomness.html
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Cargar datos
viajes_df = pd.read_pickle('dataset_viajes_raw.pkl')
print(f"N º de registros: {len(viajes_df):,}")

# Features disponibles antes del viaje
available_features = [
    'hour', 'day_of_week', 'travel_time_minutes',
    'is_morning_rush', 'is_lunch_rush', 'is_evening_rush',
    'minute_of_hour', 'minute_of_day',
    'position_ratio', 'is_early_segment', 'is_late_segment',
    'segment_number', 'total_segments',
    'is_line_3', 'is_line_5', 'is_express',
    'hour_x_position', 'rush_x_segment'
]

# Codificar variables categoricas
encoders = {}
for col in ['origin_stop_id', 'destination_stop_id', 'route_id']:
    if col in viajes_df.columns:
        le = LabelEncoder()
        viajes_df[f'{col}_encoded'] = le.fit_transform(viajes_df[col])
        encoders[col] = le
        available_features.append(f'{col}_encoded')

X = viajes_df[available_features].values
y = viajes_df['delay_at_destination'].values

print(f"Features: {len(available_features)}")
print(f"Samples: {len(X):,}")

# Split de datos: 80% Train, 20% Test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Del 80% restante: 80% train, 20% validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=42
)

print(f"\nDivision de datos:")
print(f"Train: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Val:   {len(X_val):,} ({len(X_val)/len(X)*100:.1f}%)")
print(f"Test:  {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")

# Escalar datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Funcion para calcular metricas
def calcular_metricas(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # MAPE con safe division
    epsilon = 1.0
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100
    
    # Median Absolute Error
    medae = np.median(np.abs(y_true - y_pred))
    
    # Max Error
    max_error = np.max(np.abs(y_true - y_pred))
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'medae': medae,
        'max_error': max_error
    }

# Arquitecturas de redes neuronales
#hidden_dim es tamaño de capas ocultas
# dropout: [0.1 - 0.5]  es para la regularización
# learning_rate: [0.0001 - 0.01]  es la tasa de aprendizaje (log scale)
# batch_size: [64, 128, 256] es el tamaño del batch (cantidad de muestras de datos que el modelo "ve")

class MLPSimple(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.3):
        super(MLPSimple, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.bn2 = nn.BatchNorm1d(hidden_dim//2)
        self.fc3 = nn.Linear(hidden_dim//2, 1)
        self.dropout = dropout
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc3(x)
        return x.squeeze()

class MLPMedio(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.3):
        super(MLPMedio, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.bn2 = nn.BatchNorm1d(hidden_dim//2)
        self.fc3 = nn.Linear(hidden_dim//2, hidden_dim//4)
        self.bn3 = nn.BatchNorm1d(hidden_dim//4)
        self.fc4 = nn.Linear(hidden_dim//4, 1)
        self.dropout = dropout
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc4(x)
        return x.squeeze()

class MLPProfundo(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.3):
        super(MLPProfundo, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim//2)
        self.bn3 = nn.BatchNorm1d(hidden_dim//2)
        self.fc4 = nn.Linear(hidden_dim//2, hidden_dim//4)
        self.bn4 = nn.BatchNorm1d(hidden_dim//4)
        self.fc5 = nn.Linear(hidden_dim//4, hidden_dim//8)
        self.fc6 = nn.Linear(hidden_dim//8, 1)
        self.dropout = dropout
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x.squeeze()

#aqui num_layers es el nuemero de capas apiladas
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(1, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = x.unsqueeze(-1)
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        return output.squeeze()

#n_filters -> numero de filtros convolucionales (que aplica filtros para detectar patrones automaticos)
class CNNModel(nn.Module):
    def __init__(self, input_dim, n_filters=32, dropout=0.3):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(1, n_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(n_filters, n_filters*2, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(n_filters*2, n_filters)
        self.fc2 = nn.Linear(n_filters, 1)
        self.dropout = dropout
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)

        return x.squeeze()

#d_model dimension del modelo
#nhead numero de attention heads (cada cabeza analiza los datos en paralelo)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
    
    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x.squeeze()

#Funcion para entrenar una red neuronal
def entrenar_red(model, X_tr, y_tr, X_v, y_v, epochs=100, lr=0.001, batch_size=128):
    X_tr_t = torch.FloatTensor(X_tr)
    y_tr_t = torch.FloatTensor(y_tr)
    X_v_t = torch.FloatTensor(X_v)
    y_v_t = torch.FloatTensor(y_v)
    
    train_dataset = TensorDataset(X_tr_t, y_tr_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_pred = model(X_v_t)
            val_loss = criterion(val_pred, y_v_t).item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    if best_state:
        model.load_state_dict(best_state)
    
    return best_val_loss

# Optimizacion con Optuna


results = []

# XGBoost con Optuna
print("\nXGBoost con Optuna:")

def objective_xgboost(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300), #nmero de árboles
        'max_depth': trial.suggest_int('max_depth', 4, 10), #
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0), #3muestras por árbol
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0), # features por arbol
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),  
        'gamma': trial.suggest_float('gamma', 0, 0.5),  # reduccion minima de loss
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0), #regularizacion L1
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0), #regularizacion L2
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train_scaled, y_train, verbose=False)
    pred = model.predict(X_val_scaled)
    mae = mean_absolute_error(y_val, pred)
    
    return mae

study_xgb = optuna.create_study(direction='minimize', study_name='xgboost', sampler=optuna.samplers.TPESampler(seed=42))
study_xgb.optimize(objective_xgboost, n_trials=50, show_progress_bar=True)

print(f"Mejor MAE: {study_xgb.best_value:.3f} min")

# Entrenar modelo final
start_time = time.time()
best_xgb_params = study_xgb.best_params
best_xgb_params.update({'random_state': 42, 'n_jobs': -1})
xgb_model = xgb.XGBRegressor(**best_xgb_params)
xgb_model.fit(X_train_scaled, y_train, verbose=False)
xgb_time = time.time() - start_time

# Evaluar
y_pred_test = xgb_model.predict(X_test_scaled)
metrics = calcular_metricas(y_test, y_pred_test)

print(f"Test MAE: {metrics['mae']:.3f} min ({int(metrics['mae']*60)} seg)")
print(f"Test R2:  {metrics['r2']:.4f} ({metrics['r2']*100:.1f}%)")

results.append({
    'name': 'XGBoost (Optuna)',
    'mae_test': metrics['mae'],
    'rmse_test': metrics['rmse'],
    'r2_test': metrics['r2'],
    'mape_test': metrics['mape'],
    'time': xgb_time
})

# MLP Simple con Optuna
print("\nMLP Simple (3 capas) con Optuna:")

def objective_mlp_simple(trial):
    hidden_dim = trial.suggest_int('hidden_dim', 64, 256, step=64)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    
    model = MLPSimple(len(available_features), hidden_dim, dropout)
    mae = entrenar_red(model, X_train_scaled, y_train, X_val_scaled, y_val,
                      epochs=100, lr=lr, batch_size=batch_size)
    return mae

study_mlp_s = optuna.create_study(direction='minimize', study_name='mlp_simple', sampler=optuna.samplers.TPESampler(seed=42))
study_mlp_s.optimize(objective_mlp_simple, n_trials=30, show_progress_bar=True)

print(f"Mejor MAE: {study_mlp_s.best_value:.3f} min")

# Entrenar modelo final
start_time = time.time()
best_params = study_mlp_s.best_params
model_mlp_s = MLPSimple(len(available_features),
                        best_params['hidden_dim'],
                        best_params['dropout'])
entrenar_red(model_mlp_s, X_train_scaled, y_train, X_val_scaled, y_val,
            lr=best_params['lr'], batch_size=best_params['batch_size'])
mlp_s_time = time.time() - start_time

# Evaluar
model_mlp_s.eval()
with torch.no_grad():
    y_pred_test = model_mlp_s(torch.FloatTensor(X_test_scaled)).numpy()
metrics = calcular_metricas(y_test, y_pred_test)

print(f"Test MAE: {metrics['mae']:.3f} min ({int(metrics['mae']*60)} seg)")
print(f"Test R2:  {metrics['r2']:.4f} ({metrics['r2']*100:.1f}%)")

results.append({
    'name': 'MLP Simple (3 capas)',
    'mae_test': metrics['mae'],
    'rmse_test': metrics['rmse'],
    'r2_test': metrics['r2'],
    'mape_test': metrics['mape'],
    'time': mlp_s_time
})

# MLP Medio con Optuna
print("\nMLP Medio (4 capas) con Optuna:")

def objective_mlp_medio(trial):
    hidden_dim = trial.suggest_int('hidden_dim', 128, 512, step=128)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    
    model = MLPMedio(len(available_features), hidden_dim, dropout)
    mae = entrenar_red(model, X_train_scaled, y_train, X_val_scaled, y_val,
                      epochs=100, lr=lr, batch_size=batch_size)
    return mae


study_mlp_m = optuna.create_study(direction='minimize', study_name='mlp_medio', sampler=optuna.samplers.TPESampler(seed=42))
study_mlp_m.optimize(objective_mlp_medio, n_trials=30, show_progress_bar=True)

print(f"Mejor MAE: {study_mlp_m.best_value:.3f} min")

# Entrenar modelo final
start_time = time.time()
best_params = study_mlp_m.best_params
model_mlp_m = MLPMedio(len(available_features),
                       best_params['hidden_dim'],
                       best_params['dropout'])
entrenar_red(model_mlp_m, X_train_scaled, y_train, X_val_scaled, y_val,
            lr=best_params['lr'], batch_size=best_params['batch_size'])
mlp_m_time = time.time() - start_time

# Evaluar
model_mlp_m.eval()
with torch.no_grad():
    y_pred_test = model_mlp_m(torch.FloatTensor(X_test_scaled)).numpy()
metrics = calcular_metricas(y_test, y_pred_test)

print(f"Test MAE: {metrics['mae']:.3f} min ({int(metrics['mae']*60)} seg)")
print(f"Test R2:  {metrics['r2']:.4f} ({metrics['r2']*100:.1f}%)")

results.append({
    'name': 'MLP Medio (4 capas)',
    'mae_test': metrics['mae'],
    'rmse_test': metrics['rmse'],
    'r2_test': metrics['r2'],
    'mape_test': metrics['mape'],
    'time': mlp_m_time
})

# MLP Profundo con Optuna
print("\nMLP Profundo (6 capas) con Optuna:")

def objective_mlp_profundo(trial):
    hidden_dim = trial.suggest_int('hidden_dim', 128, 512, step=128)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    
    model = MLPProfundo(len(available_features), hidden_dim, dropout)
    mae = entrenar_red(model, X_train_scaled, y_train, X_val_scaled, y_val,
                      epochs=100, lr=lr, batch_size=batch_size)
    return mae

study_mlp_p = optuna.create_study(direction='minimize', study_name='mlp_profundo', sampler=optuna.samplers.TPESampler(seed=42))
study_mlp_p.optimize(objective_mlp_profundo, n_trials=30, show_progress_bar=True)

print(f"Mejor MAE: {study_mlp_p.best_value:.3f} min")

# Entrenar modelo final
start_time = time.time()
best_params = study_mlp_p.best_params
model_mlp_p = MLPProfundo(len(available_features),
                          best_params['hidden_dim'],
                          best_params['dropout'])
entrenar_red(model_mlp_p, X_train_scaled, y_train, X_val_scaled, y_val,
            lr=best_params['lr'], batch_size=best_params['batch_size'])
mlp_p_time = time.time() - start_time

# Evaluar
model_mlp_p.eval()
with torch.no_grad():
    y_pred_test = model_mlp_p(torch.FloatTensor(X_test_scaled)).numpy()
metrics = calcular_metricas(y_test, y_pred_test)

print(f"Test MAE: {metrics['mae']:.3f} min ({int(metrics['mae']*60)} seg)")
print(f"Test R2:  {metrics['r2']:.4f} ({metrics['r2']*100:.1f}%)")

results.append({
    'name': 'MLP Profundo (6 capas)',
    'mae_test': metrics['mae'],
    'rmse_test': metrics['rmse'],
    'r2_test': metrics['r2'],
    'mape_test': metrics['mape'],
    'time': mlp_p_time
})

# LSTM con Optuna
print("\nLSTM con Optuna:")

def objective_lstm(trial):
    hidden_dim = trial.suggest_int('hidden_dim', 32, 128, step=32)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dropout = trial.suggest_float('dropout', 0.1, 0.4)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    
    model = LSTMModel(len(available_features), hidden_dim, num_layers, dropout)
    mae = entrenar_red(model, X_train_scaled, y_train, X_val_scaled, y_val,
                      epochs=100, lr=lr, batch_size=batch_size)
    return mae



study_lstm = optuna.create_study(direction='minimize', study_name='lstm', sampler=optuna.samplers.TPESampler(seed=42))
study_lstm.optimize(objective_lstm, n_trials=20, show_progress_bar=True)

print(f"Mejor MAE: {study_lstm.best_value:.3f} min")

# Entrenar modelo final
start_time = time.time()
best_params = study_lstm.best_params
model_lstm = LSTMModel(len(available_features),
                       best_params['hidden_dim'],
                       best_params['num_layers'],
                       best_params['dropout'])
entrenar_red(model_lstm, X_train_scaled, y_train, X_val_scaled, y_val,
            lr=best_params['lr'], batch_size=best_params['batch_size'])
lstm_time = time.time() - start_time

# Evaluar
model_lstm.eval()
with torch.no_grad():
    y_pred_test = model_lstm(torch.FloatTensor(X_test_scaled)).numpy()
metrics = calcular_metricas(y_test, y_pred_test)

print(f"Test MAE: {metrics['mae']:.3f} min ({int(metrics['mae']*60)} seg)")
print(f"Test R2:  {metrics['r2']:.4f} ({metrics['r2']*100:.1f}%)")

results.append({
    'name': 'LSTM',
    'mae_test': metrics['mae'],
    'rmse_test': metrics['rmse'],
    'r2_test': metrics['r2'],
    'mape_test': metrics['mape'],
    'time': lstm_time
})

# CNN con Optuna
print("\nCNN con Optuna:")

def objective_cnn(trial):
    n_filters = trial.suggest_int('n_filters', 16, 64, step=16)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    
    model = CNNModel(len(available_features), n_filters, dropout)
    mae = entrenar_red(model, X_train_scaled, y_train, X_val_scaled, y_val,
                      epochs=100, lr=lr, batch_size=batch_size)
    return mae

study_cnn = optuna.create_study(direction='minimize', study_name='cnn', sampler=optuna.samplers.TPESampler(seed=42))
study_cnn.optimize(objective_cnn, n_trials=20, show_progress_bar=True)

print(f"Mejor MAE: {study_cnn.best_value:.3f} min")

# Entrenar modelo final
start_time = time.time()
best_params = study_cnn.best_params
model_cnn = CNNModel(len(available_features),
                     best_params['n_filters'],
                     best_params['dropout'])
entrenar_red(model_cnn, X_train_scaled, y_train, X_val_scaled, y_val,
            lr=best_params['lr'], batch_size=best_params['batch_size'])
cnn_time = time.time() - start_time

# Evaluar
model_cnn.eval()
with torch.no_grad():
    y_pred_test = model_cnn(torch.FloatTensor(X_test_scaled)).numpy()
metrics = calcular_metricas(y_test, y_pred_test)

print(f"Test MAE: {metrics['mae']:.3f} min ({int(metrics['mae']*60)} seg)")
print(f"Test R2:  {metrics['r2']:.4f} ({metrics['r2']*100:.1f}%)")

results.append({
    'name': 'CNN',
    'mae_test': metrics['mae'],
    'rmse_test': metrics['rmse'],
    'r2_test': metrics['r2'],
    'mape_test': metrics['mape'],
    'time': cnn_time
})

# Transformer con Optuna
print("\nTransformer con Optuna:")

def objective_transformer(trial):
    d_model = trial.suggest_categorical('d_model', [32, 64, 128])
    nhead = trial.suggest_categorical('nhead', [2, 4, 8])
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dropout = trial.suggest_float('dropout', 0.1, 0.3)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    
    # Validar que d_model sea divisible por nhead
    if d_model % nhead != 0:
        return float('inf')
    
    model = TransformerModel(len(available_features), d_model, nhead, num_layers, dropout)
    mae = entrenar_red(model, X_train_scaled, y_train, X_val_scaled, y_val,
                      epochs=100, lr=lr, batch_size=batch_size)
    return mae

study_transformer = optuna.create_study(direction='minimize', study_name='transformer', sampler=optuna.samplers.TPESampler(seed=42))
study_transformer.optimize(objective_transformer, n_trials=20, show_progress_bar=True)

print(f"Mejor MAE: {study_transformer.best_value:.3f} min")

# Entrenar modelo final
start_time = time.time()
best_params = study_transformer.best_params
model_transformer = TransformerModel(len(available_features),
                                    best_params['d_model'],
                                    best_params['nhead'],
                                    best_params['num_layers'],
                                    best_params['dropout'])
entrenar_red(model_transformer, X_train_scaled, y_train, X_val_scaled, y_val,
            lr=best_params['lr'], batch_size=best_params['batch_size'])
transformer_time = time.time() - start_time

# Evaluar
model_transformer.eval()
with torch.no_grad():
    y_pred_test = model_transformer(torch.FloatTensor(X_test_scaled)).numpy()
metrics = calcular_metricas(y_test, y_pred_test)

print(f"Test MAE: {metrics['mae']:.3f} min ({int(metrics['mae']*60)} seg)")
print(f"Test R2:  {metrics['r2']:.4f} ({metrics['r2']*100:.1f}%)")

results.append({
    'name': 'Transformer',
    'mae_test': metrics['mae'],
    'rmse_test': metrics['rmse'],
    'r2_test': metrics['r2'],
    'mape_test': metrics['mape'],
    'time': transformer_time
})

# Resultados finales
print("\nResultados finales")

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('mae_test')

print("\n" + results_df.to_string(index=False))


results_df['rank_mae'] = results_df['mae_test'].rank(ascending=True)
results_df['rank_rmse'] = results_df['rmse_test'].rank(ascending=True)
results_df['rank_mape'] = results_df['mape_test'].rank(ascending=True)
results_df['rank_r2'] = results_df['r2_test'].rank(ascending=False)

results_df['score_final'] = results_df[['rank_mae', 'rank_rmse', 'rank_mape', 'rank_r2']].mean(axis=1)

results_df = results_df.sort_values('score_final')

best_model = results_df.iloc[0]

print(f"\nMejor modelo en general: {best_model['name']}")
print(f"MAE:  {best_model['mae_test']:.3f} (Puesto {int(best_model['rank_mae'])})")
print(f"RMSE: {best_model['rmse_test']:.3f} (Puesto {int(best_model['rank_rmse'])})")
print(f"R2:   {best_model['r2_test']:.4f} (Puesto {int(best_model['rank_r2'])})")
print(f"MAPE: {best_model['mape_test']:.1f}% (Puesto {int(best_model['rank_mape'])})")


# Visualizacion
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# MAE
ax = axes[0, 0]
models = results_df['name'].values
mae_values = results_df['mae_test'].values * 60
colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
bars = ax.bar(range(len(models)), mae_values, color=colors, alpha=0.8, edgecolor='black')
ax.set_xticks(range(len(models)))
ax.set_xticklabels(models, rotation=45, ha='right')
ax.set_ylabel('MAE Test (segundos)')
ax.set_title('Comparacion MAE')
ax.grid(True, alpha=0.3, axis='y')

for bar, mae in zip(bars, mae_values):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{int(mae)}s', ha='center', va='bottom')

# R2
ax = axes[0, 1]
r2_values = results_df['r2_test'].values
bars = ax.bar(range(len(models)), r2_values, color=colors, alpha=0.8, edgecolor='black')
ax.set_xticks(range(len(models)))
ax.set_xticklabels(models, rotation=45, ha='right')
ax.set_ylabel('R2 Test')
ax.set_title('Comparacion R2')
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3, axis='y')

for bar, r2 in zip(bars, r2_values):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{r2:.2f}', ha='center', va='bottom')

# RMSE
ax = axes[1, 0]
rmse_values = results_df['rmse_test'].values * 60
bars = ax.bar(range(len(models)), rmse_values, color=colors, alpha=0.8, edgecolor='black')
ax.set_xticks(range(len(models)))
ax.set_xticklabels(models, rotation=45, ha='right')
ax.set_ylabel('RMSE Test (segundos)')
ax.set_title('Comparacion RMSE')
ax.grid(True, alpha=0.3, axis='y')

for bar, rmse in zip(bars, rmse_values):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{int(rmse)}s', ha='center', va='bottom')

# MAPE
ax = axes[1, 1]
mape_values = results_df['mape_test'].values
bars = ax.bar(range(len(models)), mape_values, color=colors, alpha=0.8, edgecolor='black')
ax.set_xticks(range(len(models)))
ax.set_xticklabels(models, rotation=45, ha='right')
ax.set_ylabel('MAPE Test (%)')
ax.set_title('Comparacion MAPE')
ax.grid(True, alpha=0.3, axis='y')

for bar, mape in zip(bars, mape_values):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{mape:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('comparacion_modelos_completa.png', dpi=300, bbox_inches='tight')
print("Guardado: comparacion_modelos_completa.png")

