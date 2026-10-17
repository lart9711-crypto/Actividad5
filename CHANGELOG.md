### CHANGELOG.md
```markdown
# Reporte Final de Resultados - Actividad 5

## Resumen de Ejecución y Comparación de Modelos

Se evaluaron dos modelos bajo las mismas condiciones de partición de datos (80% entrenamiento, 20% prueba) y validación cruzada de 5 particiones en el conjunto de entrenamiento.

| Algoritmo | Hiperparámetros Óptimos (`GridSearchCV`) | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Regresión Logística** | `C=1.0`, `solver='lbfgs'` | 0.9737 | 0.9722 | 0.9859 | **0.9787** | 0.9912 |
| **Random Forest** | `max_depth=None`, `min_samples_split=2`, `n_estimators=100` | 0.9561 | 0.9583 | 0.9718 | 0.9650 | 0.9850 |

## Conclusiones y Decisiones
1. **Rendimiento:** La **Regresión Logística** superó ligeramente a Random Forest en este conjunto de datos linealmente separable, obteniendo un *F1-Score* de **0.9787** frente a **0.9650** de Random Forest.
2. **Trazabilidad:** Todos los parámetros, métricas y artefactos de los modelos fueron registrados exitosamente en el servidor local de **MLflow**, garantizando la reproducibilidad total del experimento.
3. **Próximos pasos:** Evaluar algoritmos de *Boosting* como XGBoost o LightGBM para verificar
