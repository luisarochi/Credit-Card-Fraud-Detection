import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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

# Paso 4: División del dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42, stratify=y
)

# Paso 5: Verificar desbalanceo en el conjunto de entrenamiento
print("Distribución antes de SMOTE:")
print(y_train.value_counts())
