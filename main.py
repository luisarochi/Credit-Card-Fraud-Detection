# mean.py

from kagglehub import KaggleDatasetAdapter, load_dataset

# Carga del dataset usando kagglehub
dataset = load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "mlg-ulb/creditcardfraud",
    "creditcard.csv"
)

# Selección de la columna 'Amount' (puedes cambiarla si necesitas otra)
amount_column = dataset["Amount"]

# Cálculo de la media
mean_amount = amount_column.mean()

# Resultado
print(f"La media de la columna 'Amount' es: {mean_amount:.2f}")
