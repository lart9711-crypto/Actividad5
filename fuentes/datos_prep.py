import os
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def cargar_y_preprocesar_datos(output_dir_ini="c:/Users/Luis/Downloads/Actividad5/datos/datos_ini", output_dir_limp="c:/Users/Luis/Downloads/Actividad5/datos/datos_limp"):
    """
    Carga el dataset público de Cáncer de Mama, guarda los datos originales 
    y aplica limpieza y escalado, guardando los datos procesados.
    """
    os.makedirs(output_dir_ini, exist_ok=True)
    os.makedirs(output_dir_limp, exist_ok=True)

    # Cargar datos públicos
    data = load_breast_cancer(as_frame=True)
    df = data.frame

    # Guardar datos originales
    path_ini = os.path.join(output_dir_ini, "cancer_original.csv")
    df.to_csv(path_ini, index=False)
    print(f"[INFO] Datos originales guardados en: {path_ini}")

    # Limpieza y preparación
    # Verificación de nulos (en este dataset no hay, pero se incluye la lógica estándar)
    df = df.dropna()

    X = df.drop(columns=["target"])
    y = df["target"]

    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Escalado de características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Reconstruir DataFrames limpios
    df_train_limp = pd.DataFrame(X_train_scaled, columns=X.columns)
    df_train_limp["target"] = y_train.values

    df_test_limp = pd.DataFrame(X_test_scaled, columns=X.columns)
    df_test_limp["target"] = y_test.values

    # Guardar datos limpios
    train_path = os.path.join(output_dir_limp, "train_limpio.csv")
    test_path = os.path.join(output_dir_limp, "test_limpio.csv")
    
    df_train_limp.to_csv(train_path, index=False)
    df_test_limp.to_csv(test_path, index=False)
    print(f"[INFO] Datos limpios guardados en: {output_dir_limp}")

    return train_path, test_path

if __name__ == "__main__":
    cargar_y_preprocesar_datos()