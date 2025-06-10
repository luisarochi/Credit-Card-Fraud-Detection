import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd

from sklearn.preprocessing import StandardScaler

# Cargar el dataset de Kaggle
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "mlg-ulb/creditcardfraud",
    "creditcard.csv"
)

print(df.head())
 
# Paso 2: Exploración de datos
print("Información del dataset:")
print(df.info())

print("\nDistribución de la variable objetivo:")
print(df['Class'].value_counts())


# Paso 3: Separar variables predictoras y objetivo
X = df.drop('Class', axis=1)
y = df['Class']

# Escalar características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
