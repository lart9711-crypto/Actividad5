# Actividad5

Este proyecto implementa un pipeline completo de Machine Learning para clasificación binaria utilizando el dataset público *Breast Cancer Wisconsin*. Compara el rendimiento de dos algoritmos de aprendizaje supervisado (**Regresión Logística** y **Random Forest**), aplicando limpieza de datos, estandarización, validación cruzada  mediante `GridSearchCV`, y seguimiento de experimentos con **MLflow**.

## Requisitos Previos
* Python 3.10 o superior
* Librerías necesarias: `scikit-learn`, `pandas`, `mlflow`, `numpy`

## Instrucciones de Ejecución
1. Clonar o posicionarse en la carpeta raíz del proyecto `Actividad5/`.
2. Ejecutar el script de preparación de datos:
   ```bash
   cd fuentes
   python datos_prep.py.
3.Ejecutar el script de entrenamiento y registro en MLflow:
Bash
python train.py.
4. Visualizar los resultados y métricas en la interfaz web de MLflow:
Bash
mlflow ui
