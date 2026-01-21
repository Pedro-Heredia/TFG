# Comparacion XGBoost vs Red Neuronal

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import time
import warnings
warnings.filterwarnings('ignore')


print("XGBoost vs Red Neuronal")
#(Solo features disponibles ANTES del viaje)


viajes_df = pd.read_pickle('dataset_viajes_raw.pkl')


# Features disponibles antes del viaje
# Estas son las que el modelo puede usar en produccion
available_features = [
    # Temporales basicas
    'hour',
    'day_of_week',
    'travel_time_minutes',
    
    # Temporales avanzadas
    'is_morning_rush',
    'is_lunch_rush',
    'is_evening_rush',
    'minute_of_hour',
    'minute_of_day',
    
    # Posicion en viaje
    'position_ratio',
    'is_early_segment',
    'is_late_segment',
    'segment_number',
    'total_segments',
    
    # Caracteristicas de linea
    'is_line_3',
    'is_line_5',
    'is_express',
    
    # Interacciones
    'hour_x_position',
    'rush_x_segment'
]

# Codificar variables categoricas
# Usamos LabelEncoder para convertir strings a numeros
#https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
encoders = {}
categorical_cols = ['origin_stop_id', 'destination_stop_id', 'route_id']

for col in categorical_cols:
    if col in viajes_df.columns:
        le = LabelEncoder()
        viajes_df[f'{col}_encoded'] = le.fit_transform(viajes_df[col])
        encoders[col] = le
        available_features.append(f'{col}_encoded')

X = viajes_df[available_features].values
y = viajes_df['delay_at_destination'].values

print(f"\nFeatures disponibles: {len(available_features)}")
print(f"Registros totales: {len(X):,}")

# Split 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nDivision de datos:")
print(f"Train: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Test:  {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")

# Escalar datos
# StandardScaler para normalizar cada feature a media=0 y std=1
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Estadisticas target
print(f"\nEstadisticas de delay_at_destination:")
print(f"Media: {y.mean():.2f} min")
print(f"Std:   {y.std():.2f} min")
print(f"Min:   {y.min():.2f} min")
print(f"Max:   {y.max():.2f} min")

#XGBoost
print("\n")
print("MODELO 1: XGBoost")


print("Entrenando XGBoost")
start_time = time.time()

# Configuracion de XGBoost
#  https://xgboost.readthedocs.io/en/stable/parameter.html
xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1,
    random_state=42,
    n_jobs=-1
)

xgb_model.set_params(early_stopping_rounds=20)
xgb_model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_test_scaled, y_test)],
    verbose=False
)

xgb_train_time = time.time() - start_time

# Predicciones
y_pred_xgb_train = xgb_model.predict(X_train_scaled)
y_pred_xgb_test = xgb_model.predict(X_test_scaled)

# Metricas
xgb_mae_train = mean_absolute_error(y_train, y_pred_xgb_train)
xgb_mae_test = mean_absolute_error(y_test, y_pred_xgb_test)
xgb_rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_xgb_train))
xgb_rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_xgb_test))
xgb_r2_train = r2_score(y_train, y_pred_xgb_train)
xgb_r2_test = r2_score(y_test, y_pred_xgb_test)

print(f"\nXGBoost entrenado en {xgb_train_time:.2f}s")
print(f"Metricas XGBoost:")
print(f"\n\tTrain:")
print(f"MAE:  {xgb_mae_train:.3f} min ({xgb_mae_train*60:.0f} seg)")
print(f"RMSE: {xgb_rmse_train:.3f} min")
print(f"R2:   {xgb_r2_train:.4f} ({xgb_r2_train*100:.2f}%)")
print(f"\n\tTest:")
print(f"MAE:  {xgb_mae_test:.3f} min ({xgb_mae_test*60:.0f} seg)")
print(f"RMSE: {xgb_rmse_test:.3f} min")
print(f"R2:   {xgb_r2_test:.4f} ({xgb_r2_test*100:.2f}%)")

# Feature importance (para saber para este modelo, que es le es lo mas importante para calcular el retraso)
feature_importance = pd.DataFrame({
    'feature': available_features,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 10 Features mas importantes:")
for idx, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']:30s}: {row['importance']:.4f} ({row['importance']*100:.1f}%)")






# MODELO 2: Red Neuronal (MLP)
print("\nMODELO 2: Red Neuronal (MLP)")


# arquitectura de red neuronal (6 capas)
#https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html
# aqui cada capa analiza los datos y extrae caracteristicas importantes
class ImprovedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.3):
        super(ImprovedMLP, self).__init__()
        # Capas fully connected
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        # Capa de atencion
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn4 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc5 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc6 = nn.Linear(hidden_dim // 4, 1)
        self.dropout = dropout
    
    def forward(self, x):
        # Primera capa
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Segunda capa
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Tercera capa
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Mecanismo de atencion. Droupout para que no haya overfitting
        attention_weights = torch.sigmoid(self.attention(x))
        x = x * attention_weights
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Capas finales
        x = self.fc4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x.squeeze()

# Preparar datos para PyTorch
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Inicializar modelo
mlp_model = ImprovedMLP(input_dim=len(available_features), hidden_dim=256, dropout=0.3)
criterion = nn.L1Loss()  # MAE loss
optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.001, weight_decay=1e-5)


#(BatchNorm + Dropout(0.3) en cada capa)

print("\nEntrenando Red Neuronal:")
start_time = time.time()

epochs = 100 #ilmite de intentos
best_test_loss = float('inf') # record a batir (infinito obviamente)
patience = 15 # margen de espera
patience_counter = 0 # contador de fallos

for epoch in range(epochs):
    # Entrenamiento
    mlp_model.train()
    train_loss = 0
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = mlp_model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validacion
    mlp_model.eval()
    with torch.no_grad():
        test_pred = mlp_model(X_test_tensor)
        test_loss = criterion(test_pred, y_test_tensor).item()
    
    # Early stopping (si despues de un rato no mejoran los datos, pues paramos)
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        patience_counter = 0
        best_model_state = mlp_model.state_dict()
    else:
        patience_counter += 1
    
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_loader):.4f} - Test Loss: {test_loss:.4f}")
    
    if patience_counter >= patience:
        print(f"  Early stopping en epoch {epoch+1}")
        break

mlp_model.load_state_dict(best_model_state)
mlp_train_time = time.time() - start_time

# Predicciones
mlp_model.eval()
with torch.no_grad():
    y_pred_mlp_train = mlp_model(X_train_tensor).numpy()
    y_pred_mlp_test = mlp_model(X_test_tensor).numpy()

# Metricas
mlp_mae_train = mean_absolute_error(y_train, y_pred_mlp_train)
mlp_mae_test = mean_absolute_error(y_test, y_pred_mlp_test)
mlp_rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_mlp_train))
mlp_rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_mlp_test))
mlp_r2_train = r2_score(y_train, y_pred_mlp_train)
mlp_r2_test = r2_score(y_test, y_pred_mlp_test)

print(f"\nRed Neuronal entrenada en {mlp_train_time:.2f}s")
print(f"Metricas Red Neuronal:")
print(f"\n\tTrain:")
print(f"MAE:  {mlp_mae_train:.3f} min ({mlp_mae_train*60:.0f} seg)")
print(f"RMSE: {mlp_rmse_train:.3f} min")
print(f"R2:   {mlp_r2_train:.4f} ({mlp_r2_train*100:.2f}%)")
print(f"\n\tTest:")
print(f"MAE:  {mlp_mae_test:.3f} min ({mlp_mae_test*60:.0f} seg)")
print(f"RMSE: {mlp_rmse_test:.3f} min")
print(f"R2:   {mlp_r2_test:.4f} ({mlp_r2_test*100:.2f}%)")




#Comparacion final

print("\nComparacion")


comparison = pd.DataFrame({
    'Modelo': ['XGBoost', 'Red Neuronal (MLP)'],
    'MAE Train': [f"{xgb_mae_train:.3f}", f"{mlp_mae_train:.3f}"],
    'MAE Test': [f"{xgb_mae_test:.3f}", f"{mlp_mae_test:.3f}"],
    'RMSE Test': [f"{xgb_rmse_test:.3f}", f"{mlp_rmse_test:.3f}"],
    'R2 Test': [f"{xgb_r2_test:.4f}", f"{mlp_r2_test:.4f}"],
    'Tiempo (s)': [f"{xgb_train_time:.1f}", f"{mlp_train_time:.1f}"]
})

print("\n" + comparison.to_string(index=False))

# Determinar ganador
if mlp_mae_test < xgb_mae_test:
    winner = "Red Neuronal"
    mae_diff = ((xgb_mae_test - mlp_mae_test) / xgb_mae_test) * 100
    r2_diff = ((mlp_r2_test - xgb_r2_test) / xgb_r2_test) * 100
    print(f"\nGanador: {winner}")
    print(f"MAE: {mae_diff:+.1f}% mejor que XGBoost")
    print(f"R2:  {r2_diff:+.1f}% mejor que XGBoost")
else:
    winner = "XGBoost"
    mae_diff = ((mlp_mae_test - xgb_mae_test) / mlp_mae_test) * 100
    r2_diff = ((xgb_r2_test - mlp_r2_test) / mlp_r2_test) * 100
    print(f"\nGanador: {winner}")
    print(f"MAE: {mae_diff:+.1f}% mejor que Red Neuronal")
    print(f"R2:  {r2_diff:+.1f}% mejor que Red Neuronal")






# VISUALIZACIONES

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Comparacion MAE
ax = axes[0, 0]
models = ['XGBoost', 'Red Neuronal']
mae_values = [xgb_mae_test, mlp_mae_test]
colors = ['#ff7f0e', '#2ca02c']
bars = ax.bar(models, mae_values, color=colors, alpha=0.7)
ax.set_ylabel('MAE (minutos)', fontsize=12)
ax.set_title('Comparacion MAE - Prediccion SIN delay actual', fontsize=14, fontweight='bold')
ax.set_ylim(0, max(mae_values) * 1.2)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f} min\n({height*60:.0f} seg)', 
            ha='center', va='bottom', fontsize=10)

# 2. Predicciones vs Reales - XGBoost
ax = axes[0, 1]
sample_size = min(1000, len(y_test))
indices = np.random.choice(len(y_test), sample_size, replace=False)
ax.scatter(y_test[indices], y_pred_xgb_test[indices], alpha=0.3, s=10, color='#ff7f0e')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel('Delay Real (min)', fontsize=11)
ax.set_ylabel('Delay Predicho (min)', fontsize=11)
ax.set_title(f'XGBoost\nMAE: {xgb_mae_test:.3f} min ({xgb_mae_test*60:.0f} seg)', 
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# 3. Predicciones vs Reales - Red Neuronal
ax = axes[1, 0]
ax.scatter(y_test[indices], y_pred_mlp_test[indices], alpha=0.3, s=10, color='#2ca02c')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel('Delay Real (min)', fontsize=11)
ax.set_ylabel('Delay Predicho (min)', fontsize=11)
ax.set_title(f'Red Neuronal\nMAE: {mlp_mae_test:.3f} min ({mlp_mae_test*60:.0f} seg)', 
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# 4. Feature Importance (XGBoost)
ax = axes[1, 1]
top_features = feature_importance.head(10)
ax.barh(range(len(top_features)), top_features['importance'], color='#ff7f0e', alpha=0.7)
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['feature'], fontsize=9)
ax.set_xlabel('Importancia', fontsize=11)
ax.set_title('Top 10 Features (XGBoost)', fontsize=12, fontweight='bold')
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('comparacion_entrenamiento_correcto.png', dpi=300, bbox_inches='tight')




# Guardar modelos

torch.save({
    'model_state': mlp_model.state_dict(),
    'scaler': scaler,
    'encoders': encoders,
    'features': available_features,
    'mae': mlp_mae_test
}, 'modelo_red_neuronal_correcto.pth')

import pickle
with open('modelo_xgboost_correcto.pkl', 'wb') as f:
    pickle.dump({
        'model': xgb_model,
        'scaler': scaler,
        'encoders': encoders,
        'features': available_features,
        'mae': xgb_mae_test
    }, f)

print("Modelos generados:")
print("modelo_red_neuronal_correcto.pth")
print("modelo_xgboost_correcto.pkl")
print("comparacion_entrenamiento_correcto.png")