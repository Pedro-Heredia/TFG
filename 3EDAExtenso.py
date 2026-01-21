#Evaluacion de Datos (EDA)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Configuracion de visualizacion
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10


#carga de datos

viajes_df = pd.read_pickle('dataset_viajes_raw.pkl')

print(f"\nDataset original:")
print(f"Total registros: {len(viajes_df):,}")
print(f"Columnas: {viajes_df.shape[1]}")



initial_size = len(viajes_df)

'''
# Eliminar registros con NULL en campos criticos
critical_columns = [
    'travel_time_minutes', 'delay_at_destination', 'delay_at_origin',
    'origin_stop_id', 'destination_stop_id', 'route_id',
    'hour', 'day_of_week'
]

for col in critical_columns:
    if col in viajes_df.columns:
        viajes_df = viajes_df[viajes_df[col].notna()]

print(f"Eliminados por NULL: {initial_size - len(viajes_df):,}")

# Eliminar travel_time fisicamente imposible (< 30 segundos)
before = len(viajes_df)
viajes_df = viajes_df[viajes_df['travel_time_minutes'] >= 0.5]
print(f"Eliminados por travel_time < 0.5 min: {before - len(viajes_df):,}")'''

print(f"Registros validos: {len(viajes_df):,}")
print(f"Datos retenidos: {len(viajes_df)/initial_size*100:.1f}%")


#estadisticas

print(f"\nVariable objetivo: delay_at_destination")
print(f"Media:        {viajes_df['delay_at_destination'].mean():.2f} min")
print(f"Mediana:      {viajes_df['delay_at_destination'].median():.2f} min")
print(f"Desv. Std:    {viajes_df['delay_at_destination'].std():.2f} min")
print(f"Minimo:       {viajes_df['delay_at_destination'].min():.2f} min")
print(f"Maximo:       {viajes_df['delay_at_destination'].max():.2f} min")
print(f"Q25:          {viajes_df['delay_at_destination'].quantile(0.25):.2f} min") #25% de los viajes tienen un retraso igual o menos a este vfalor
print(f"Q75:          {viajes_df['delay_at_destination'].quantile(0.75):.2f} min") #75% de los viajes tienen un retraso igual o menos a este vfalor
print(f"Q90:          {viajes_df['delay_at_destination'].quantile(0.90):.2f} min")
print(f"Q95:          {viajes_df['delay_at_destination'].quantile(0.95):.2f} min")
print(f"Q99:          {viajes_df['delay_at_destination'].quantile(0.99):.2f} min")


#Distribucion de delays

adelantos = (viajes_df['delay_at_destination'] < -0.5).sum()
puntuales = ((viajes_df['delay_at_destination'] >= -0.5) & 
             (viajes_df['delay_at_destination'] <= 0.5)).sum()
retrasos_leves = ((viajes_df['delay_at_destination'] > 0.5) & 
                  (viajes_df['delay_at_destination'] <= 3)).sum()
retrasos_graves = (viajes_df['delay_at_destination'] > 3).sum()

total = len(viajes_df)
print(f"\nDistribucion de delays:")
print(f"Adelantos (< -0.5 min):     {adelantos:>6,} ({adelantos/total*100:>5.1f}%)")
print(f"Puntuales (-0.5 a +0.5):    {puntuales:>6,} ({puntuales/total*100:>5.1f}%)")
print(f"Retrasos leves (+0.5 a +3): {retrasos_leves:>6,} ({retrasos_leves/total*100:>5.1f}%)")
print(f"Retrasos graves (> +3):     {retrasos_graves:>6,} ({retrasos_graves/total*100:>5.1f}%)")


# Test de normalidad (Shapiro-Wilk)
# Usamos una muestra porque Shapiro-Wilk tiene limite de 5000 registros (sino peta)
sample_size = min(5000, len(viajes_df))
sample_delays = viajes_df['delay_at_destination'].sample(sample_size, random_state=42)
stat_shapiro, p_shapiro = stats.shapiro(sample_delays)

print(f"\nTest de Normalidad (Shapiro-Wilk):")
print(f"Estadistico: {stat_shapiro:.4f}")
print(f"p-value:     {p_shapiro:.6f}")
if p_shapiro > 0.05:
    print(f"Conclusion: Distribucion NORMAL (alpha=0.05)")
else:
    print(f"Conclusion: Distribucion NO NORMAL (alpha=0.05)")



#Analisis por linea 

print("\nAnalisis por lineas")


# Agrupar por linea y calcular estadisticas
line_stats = viajes_df.groupby('route_id').agg({
    'delay_at_destination': ['mean', 'median', 'std', 'min', 'max', 'count'],
    'travel_time_minutes': ['mean', 'median']
}).round(2)

line_stats.columns = ['Delay_Mean', 'Delay_Median', 'Delay_Std', 
                      'Delay_Min', 'Delay_Max', 'N_Viajes',
                      'TravelTime_Mean', 'TravelTime_Median']

line_stats = line_stats.sort_values('Delay_Mean', ascending=False)

print(f"\nEstadisticas por linea:")
print(line_stats.to_string())

print(f"\nTop 3 lineas con MAS retraso:")
for i, (line, row) in enumerate(line_stats.head(3).iterrows(), 1):
    print(f"{i}. Linea {line}: {row['Delay_Mean']:.2f} min promedio ({row['N_Viajes']:,} viajes)")

print(f"\nTop 3 lineas MAS puntuales:")
for i, (line, row) in enumerate(line_stats.tail(3).iterrows(), 1):
    print(f"{i}. Linea {line}: {row['Delay_Mean']:.2f} min promedio ({row['N_Viajes']:,} viajes)")


# Test ANOVA: comparar si hay diferencias significativas entre lineas
# Extraer delays de cada linea como listas separadas
lines_delays = [group['delay_at_destination'].values 
                for _, group in viajes_df.groupby('route_id')]
f_stat, p_anova = stats.f_oneway(*lines_delays)

print(f"\nTest ANOVA (diferencia entre lineas):")
print(f"F-estadistico: {f_stat:.2f}")
print(f"p-value:       {p_anova:.10f}")
if p_anova < 0.05:
    print(f"Conclusion: SI hay diferencia significativa (alpha=0.05)")
else:
    print(f"Conclusion: NO hay diferencia significativa (alpha=0.05)")


#Analisis temporal -> hora del dia

# Agrupar por hora
hour_stats = viajes_df.groupby('hour').agg({
    'delay_at_destination': ['mean', 'std', 'count']
}).round(2)

hour_stats.columns = ['Delay_Mean', 'Delay_Std', 'N_Viajes']

print(f"\nRetraso promedio por hora:")
print(hour_stats.to_string())

# Identificar horas pico (retraso mayor al promedio)
peak_hours = hour_stats[hour_stats['Delay_Mean'] > hour_stats['Delay_Mean'].mean()]
print(f"\nHoras con MAYOR retraso  promedio:")
for hour in peak_hours.index:
    print(f"{hour:02d}:00 -> {peak_hours.loc[hour, 'Delay_Mean']:.2f} min ({peak_hours.loc[hour, 'N_Viajes']:,} viajes)")

# Correlacion hora vs delay (Pearson)
#(a medida que pasa el dia, los retrasos aumentan o disminuyen)
corr_hour, p_hour = pearsonr(viajes_df['hour'], viajes_df['delay_at_destination'])
print(f"\nCorrelacion hora vs delay:")
print(f"Pearson r: {corr_hour:.4f}")
print(f"p-value:   {p_hour:.10f}")
if p_hour < 0.05:
    print(f"Conclusion: Correlacion significativa")
else:
    print(f"Conclusion: No hay correlacion significativa")


#Analisis dia de la semana


# Mapear numeros a nombres de dias
day_names = {0: 'Lunes', 1: 'Martes', 2: 'Miercoles', 3: 'Jueves', 
             4: 'Viernes', 5: 'Sabado', 6: 'Domingo'}

viajes_df['day_name'] = viajes_df['day_of_week'].map(day_names)

# Agrupar por dia
dow_stats = viajes_df.groupby('day_name').agg({
    'delay_at_destination': ['mean', 'std', 'count']
}).round(2)

dow_stats.columns = ['Delay_Mean', 'Delay_Std', 'N_Viajes']

# Ordenar por dia de semana
dow_order = ['Lunes', 'Martes', 'Miercoles', 'Jueves', 'Viernes', 'Sabado', 'Domingo']
dow_stats = dow_stats.reindex(dow_order)

print(f"\nRetraso promedio por dia:")
print(dow_stats.to_string())

# Comparar laborables vs fin de semana (t-test)
laborables = viajes_df[viajes_df['day_of_week'] < 5]['delay_at_destination']
finde = viajes_df[viajes_df['day_of_week'] >= 5]['delay_at_destination']

t_stat, p_ttest = stats.ttest_ind(laborables, finde)

print(f"\nTest t-Student (laborables vs fin de semana):")
print(f"Laborables:     {laborables.mean():.2f} +/- {laborables.std():.2f} min")
print(f"Fin de semana:  {finde.mean():.2f} +/- {finde.std():.2f} min")
print(f"t-estadistico:  {t_stat:.4f}")
print(f"p-value:        {p_ttest:.10f}")
if p_ttest < 0.05:
    print(f"Conclusion: Diferencia significativa")
else:
    print(f"Conclusion: No hay diferencia significativa")


#correlacion entre variables

# Variables numericas disponibles
numeric_vars = ['travel_time_minutes', 'delay_at_origin', 'cumulative_delay_origin',
                'hour', 'day_of_week', 'segment_number', 'total_segments', 
                'position_ratio']

# Filtrar solo las que existen
numeric_vars = [v for v in numeric_vars if v in viajes_df.columns]

# Calcular matriz de correlacion
corr_matrix = viajes_df[numeric_vars + ['delay_at_destination']].corr()

print(f"\nCorrelacion con delay_at_destination:")
correlations = corr_matrix['delay_at_destination'].drop('delay_at_destination').sort_values(ascending=False)

# Mostrar correlaciones con su significancia estadistica
for var, corr in correlations.items():
    # Calcular p-value para cada correlacion
    _, p_val = pearsonr(viajes_df[var], viajes_df['delay_at_destination'])
    
    # Determinar significancia
    if p_val < 0.001:
        sig = "***"
    elif p_val < 0.01:
        sig = "**"
    elif p_val < 0.05:
        sig = "*"
    else:
        sig = "ns"
    
    print(f"{var:30s}: r = {corr:+.4f}  (p < {p_val:.6f}) {sig}")

print(f"\nSignificancia: *** p<0.001, ** p<0.01, * p<0.05, ns = no significativo")

#Basicamente filtro de importancia, para saber si las variables tienen relacion real con los retrasos o son ruidos



#Analisis trenes que saltan paradas
print("Analisis trenes que saltan paradas")

# Agrupar por train_id y contar paradas
train_stops = viajes_df.groupby('train_id').agg({
    'segment_number': 'max',
    'route_id': 'first'
}).rename(columns={'segment_number': 'num_segments'})

# Calcular estadisticas de paradas por linea
line_avg_stops = train_stops.groupby('route_id')['num_segments'].agg(['mean', 'std', 'min', 'max'])

print(f"\nNumero de paradas por linea:")
print(line_avg_stops.round(1).to_string())

# Identificar trenes express (menos paradas que el promedio)
for line in line_avg_stops.index:
    avg = line_avg_stops.loc[line, 'mean']
    std = line_avg_stops.loc[line, 'std']
    line_trains = train_stops[train_stops['route_id'] == line]
    
    # Trenes con mas de 1 desviacion estandar menos paradas
    express_trains = line_trains[line_trains['num_segments'] < (avg - std)]
    
    if len(express_trains) > 0:
        pct = len(express_trains) / len(line_trains) * 100
        print(f"\nLinea {line}:")
        print(f"Promedio paradas: {avg:.1f}")
        print(f"Trenes express/limitados: {len(express_trains)} ({pct:.1f}%)")
        print(f"Paradas minimas: {line_trains['num_segments'].min()}")

#distinción entre servicio Local (para en todas) y Express (se salta paradas)


#Analisis no linealidad
print("Analisis no linealidad")

# Tomar muestra para analisis (10000 registros o todos si hay menos)
sample = viajes_df.sample(min(10000, len(viajes_df)), random_state=42)

# Comparar correlacion lineal (Pearson) vs no lineal (Spearman)
pearson_r, p_pearson = pearsonr(sample['travel_time_minutes'], sample['delay_at_destination'])
spearman_r, p_spearman = spearmanr(sample['travel_time_minutes'], sample['delay_at_destination'])

print(f"\nRelacion travel_time vs delay:")
print(f"Pearson (lineal):    r = {pearson_r:.4f}, p = {p_pearson:.6f}")
print(f"Spearman (no lineal): rho = {spearman_r:.4f}, p = {p_spearman:.6f}")

# Si la diferencia es grande, hay no linealidad
if abs(spearman_r - pearson_r) > 0.1:
    print(f"Conclusion: Relacion NO LINEAL detectada (diferencia > 0.1)")
else:
    print(f"Conclusion: Relacion aproximadamente lineal")






# Graficas


fig = plt.figure(figsize=(20, 15))
gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

# 1. Distribucion de delays
ax1 = fig.add_subplot(gs[0, 0])
viajes_df['delay_at_destination'].hist(bins=100, ax=ax1, edgecolor='black', alpha=0.7)
ax1.axvline(viajes_df['delay_at_destination'].mean(), color='red', linestyle='--', 
            label=f'Media: {viajes_df["delay_at_destination"].mean():.2f}', linewidth=2)
ax1.axvline(viajes_df['delay_at_destination'].median(), color='green', linestyle='--',
            label=f'Mediana: {viajes_df["delay_at_destination"].median():.2f}', linewidth=2)
ax1.set_xlabel('Delay (min)')
ax1.set_ylabel('Frecuencia')
ax1.set_title('Distribucion de Delays (TODOS los datos)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Boxplot por linea
ax2 = fig.add_subplot(gs[0, 1])
viajes_df.boxplot(column='delay_at_destination', by='route_id', ax=ax2)
ax2.set_xlabel('Linea')
ax2.set_ylabel('Delay (min)')
ax2.set_title('Distribucion de Delays por Linea')
plt.sca(ax2)
plt.xticks(rotation=0)

# 3. Delays por hora
ax3 = fig.add_subplot(gs[0, 2])
hour_stats['Delay_Mean'].plot(kind='bar', ax=ax3, color='steelblue', alpha=0.7, edgecolor='black')
ax3.axhline(viajes_df['delay_at_destination'].mean(), color='red', linestyle='--', linewidth=2)
ax3.set_xlabel('Hora del Dia')
ax3.set_ylabel('Delay Promedio (min)')
ax3.set_title('Retraso Promedio por Hora')
ax3.grid(axis='y', alpha=0.3)

# 4. Delays por dia de semana
ax4 = fig.add_subplot(gs[1, 0])
dow_stats['Delay_Mean'].plot(kind='bar', ax=ax4, color='coral', alpha=0.7, edgecolor='black')
ax4.set_xlabel('Dia de la Semana')
ax4.set_ylabel('Delay Promedio (min)')
ax4.set_title('Retraso Promedio por Dia')
ax4.set_xticklabels(dow_stats.index, rotation=45)
ax4.grid(axis='y', alpha=0.3)

# 5. Scatter: travel_time vs delay
ax5 = fig.add_subplot(gs[1, 1])
sample_plot = viajes_df.sample(min(5000, len(viajes_df)), random_state=42)
ax5.scatter(sample_plot['travel_time_minutes'], sample_plot['delay_at_destination'], 
            alpha=0.3, s=10)
ax5.set_xlabel('Travel Time (min)')
ax5.set_ylabel('Delay (min)')
ax5.set_title(f'Travel Time vs Delay (Pearson r={pearson_r:.3f})')
ax5.grid(True, alpha=0.3)

# 6. Matriz de correlacion
ax6 = fig.add_subplot(gs[1, 2])
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            ax=ax6, cbar_kws={'label': 'Correlacion'})
ax6.set_title('Matriz de Correlaciones')

# 7. Q-Q plot (test de normalidad)
ax7 = fig.add_subplot(gs[2, 0])
stats.probplot(viajes_df['delay_at_destination'].sample(min(5000, len(viajes_df)), random_state=42), 
               dist="norm", plot=ax7)
ax7.set_title('Q-Q Plot (Test de Normalidad)')
ax7.grid(True, alpha=0.3)

# 8. Delays por posicion en viaje
ax8 = fig.add_subplot(gs[2, 1])
position_bins = pd.cut(viajes_df['position_ratio'], bins=[0, 0.33, 0.67, 1.0],
                       labels=['Inicio', 'Medio', 'Final'])
viajes_df.groupby(position_bins)['delay_at_destination'].mean().plot(
    kind='bar', ax=ax8, color='purple', alpha=0.7, edgecolor='black')
ax8.set_xlabel('Posicion en el Viaje')
ax8.set_ylabel('Delay Promedio (min)')
ax8.set_title('Retraso por Posicion en el Viaje')
ax8.set_xticklabels(['Inicio', 'Medio', 'Final'], rotation=0)
ax8.grid(axis='y', alpha=0.3)

# 9. Numero de paradas por linea
ax9 = fig.add_subplot(gs[2, 2])
line_avg_stops['mean'].sort_values(ascending=False).plot(
    kind='barh', ax=ax9, color='teal', alpha=0.7, edgecolor='black')
ax9.set_xlabel('Numero Promedio de Paradas')
ax9.set_ylabel('Linea')
ax9.set_title('Paradas Promedio por Linea')
ax9.grid(axis='x', alpha=0.3)

# 10. Percentiles de delay
ax10 = fig.add_subplot(gs[3, 0])
percentiles = [10, 25, 50, 75, 90, 95, 99]
percentile_values = [viajes_df['delay_at_destination'].quantile(p/100) for p in percentiles]
ax10.bar([f'P{p}' for p in percentiles], percentile_values, 
         color='orange', alpha=0.7, edgecolor='black')
ax10.set_ylabel('Delay (min)')
ax10.set_title('Percentiles de Delay')
ax10.grid(axis='y', alpha=0.3)

# 11. CDF (Cumulative Distribution Function)
ax11 = fig.add_subplot(gs[3, 1])
sorted_delays = np.sort(viajes_df['delay_at_destination'])
cdf = np.arange(1, len(sorted_delays)+1) / len(sorted_delays)
ax11.plot(sorted_delays, cdf, linewidth=2)
ax11.axvline(0, color='red', linestyle='--', alpha=0.5)
ax11.set_xlabel('Delay (min)')
ax11.set_ylabel('Probabilidad Acumulada')
ax11.set_title('CDF de Delays')
ax11.grid(True, alpha=0.3)

# 12. Violinplot top 5 lineas
ax12 = fig.add_subplot(gs[3, 2])
top_lines = line_stats.head(5).index
violin_data = [viajes_df[viajes_df['route_id'] == line]['delay_at_destination'].values 
               for line in top_lines]
ax12.violinplot(violin_data, positions=range(len(top_lines)), showmeans=True)
ax12.set_xticks(range(len(top_lines)))
ax12.set_xticklabels(top_lines)
ax12.set_ylabel('Delay (min)')
ax12.set_title('Distribucion de Delays - Top 5 Lineas')
ax12.grid(axis='y', alpha=0.3)

plt.savefig('EDA_completo.png', dpi=300, bbox_inches='tight')
print(f"\nGuardado: EDA_completo.png")


#Conclusiones y recomendaciones de modelo
print("conclusiones y recomendacion de modelo") 

# Determinar tipo de relacion
if p_anova < 0.05 and abs(spearman_r - pearson_r) > 0.1:
    modelo_recomendado = "MODELOS NO LINEALES (Random Forest, XGBoost, Deep Learning)"
    justificacion = """
Justificacion:
1. Relacion NO LINEAL detectada (Spearman vs Pearson)
2. Diferencias significativas entre lineas (ANOVA)
3. Interacciones complejas entre variables
4. Presencia de outliers y distribucion no normal
"""
elif p_anova < 0.05:
    modelo_recomendado = "MODELOS ENSEMBLE (Random Forest, XGBoost)"
    justificacion = """
Justificacion:
1. Diferencias significativas entre lineas
2. Multiples variables correlacionadas
3. Relacion aproximadamente lineal pero compleja
"""
else:
    modelo_recomendado = "MODELOS SIMPLES (Regresion Lineal, Ridge, Lasso)"
    justificacion = """
Justificacion:
1. Relaciones aproximadamente lineales
2. No hay diferencias dramaticas entre grupos
3. Datos relativamente limpios
"""

print(f"\nModelo recomendado: {modelo_recomendado}")
print(justificacion)

print(f"\nCaracteristicas de los datos:")
print(f"Total viajes validos: {len(viajes_df):,}")
print(f"Rango de delays: {viajes_df['delay_at_destination'].min():.1f} a {viajes_df['delay_at_destination'].max():.1f} min")
print(f"Distribucion: {'NO NORMAL' if p_shapiro < 0.05 else 'NORMAL'} (Shapiro p={p_shapiro:.6f})")
#Presencia de outliers: Si (delays > 3 std)

print(f"\nRelaciones detectadas:")
print(f"Correlacion hora vs delay: r={corr_hour:.3f} (p={p_hour:.6f})")
print(f"Diferencia entre lineas: {'SI significativa' if p_anova < 0.05 else 'NO significativa'} (ANOVA p={p_anova:.6f})")
print(f"Diferencia lab/finde: {'SI significativa' if p_ttest < 0.05 else 'NO significativa'} (t-test p={p_ttest:.6f})")
print(f"No linealidad: {'DETECTADA' if abs(spearman_r - pearson_r) > 0.1 else 'NO detectada'}")

print(f"\nComplejidad del problema:")
print(f"Variables correlacionadas: {(abs(correlations) > 0.3).sum()} de {len(correlations)}")
#Interacciones evidentes: Hora x Linea, Posicion x Delay previo
#Dependencia temporal: Si (delay_at_origin correlaciona con delay_at_destination)

