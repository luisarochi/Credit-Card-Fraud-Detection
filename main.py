import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    roc_auc_score,
    classification_report
)
from imblearn.over_sampling import SMOTE


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

# Paso 1: Separar variables y target
X = df.drop('Class', axis=1)
y = df['Class']

# Paso 2: Dividir en train/test (estratificado para mantener proporción)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Paso 3: Aplicar SMOTE SOLO en entrenamiento
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Luego, entrenar el modelo con X_train_smote, y_train_smote
# y evaluar en X_test, y_test

# Paso 6: Aplicación de SMOTE y entrenamiento del modelo

# Aplicar SMOTE para balancear las clases
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Verificar la nueva distribución
print("Distribución después de SMOTE:")
print(y_train_smote.value_counts())

# Entrenar el modelo
model = RandomForestClassifier(random_state=42)
model.fit(X_train_smote, y_train_smote)

# Evaluar el modelo
y_pred = model.predict(X_test)

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

# Paso 7: Evaluación avanzada del modelo con métricas para datos desbalanceados
#Predecir etiquetas
y_pred = model.predict(X_test)

# Predecir probabilidades para clase positiva (para AUC)
y_proba = model.predict_proba(X_test)[:, 1]

# Mostrar reporte completo
print("Reporte de clasificación:\n", classification_report(y_test, y_pred))

print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))

print("AUC-ROC:", roc_auc_score(y_test, y_proba))

# Paso 8: Entrenar modelo con datos balanceados y evaluar

# Instanciar el modelo (puedes usar LogisticRegression u otro)
model = LogisticRegression(random_state=42, max_iter=1000)

# Entrenar con los datos balanceados
model.fit(X_train_smote, y_train_smote)

# Predecir con el conjunto de prueba original (sin balancear)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Evaluar el modelo
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_pred_proba))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# Predicciones en test
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Métricas
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_pred_proba))
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))

# Curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc_score(y_test, y_pred_proba))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

params = {'n_estimators': [50, 100], 'max_depth': [10, 20]}
grid = GridSearchCV(model, param_grid=params, scoring='roc_auc', cv=3)
grid.fit(X_train_smote, y_train_smote)

print("Mejores parámetros:", grid.best_params_)
print("Mejor score ROC AUC:", grid.best_score_)


joblib.dump(model, 'modelo_credit_fraud.pkl')