import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def entrenar_y_registrar():
    # Configurar experimento en MLflow
    mlflow.set_experiment("Actividad5_Clasificacion_MLflow")

    # Cargar datos limpios
    train_df = pd.read_csv("c:/Users/Luis/Downloads/Actividad5/datos/datos_limp/train_limpio.csv")
    test_df = pd.read_csv("c:/Users/Luis/Downloads/Actividad5/datos/datos_limp/test_limpio.csv")

    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    # Definir modelos y espacios de hiperparámetros para Grid Search
    modelos = {
        "LogisticRegression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "params": {
                "C": [0.01, 0.1, 1.0, 10.0],
                "solver": ["lbfgs"]
            }
        },
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 5, 10],
                "min_samples_split": [2, 5]
            }
        }
    }

    for nombre, config in modelos.items():
        print(f"\n[INFO] Entrenando y optimizando: {nombre}...")
        
        with mlflow.start_run(run_name=nombre):
            # GridSearchCV (K-Fold con 5 particiones)
            grid_search = GridSearchCV(
                estimator=config["model"],
                param_grid=config["params"],
                cv=5,
                scoring="f1",
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Mejor modelo encontrado
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            
            # Predicciones y métricas en conjunto de prueba
            y_pred = best_model.predict(X_test)
            y_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else y_pred

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_proba)

            mlflow.log_params(best_params)
            mlflow.log_param("algoritmo", nombre)

            # Registrar métricas estandarizadas
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", roc_auc)

           
            mlflow.sklearn.log_model(best_model, artifact_path=f"modelo_{nombre}",skops_trusted_types=["sklearn.tree._tree.Tree"])
            
            print(f"[EXITO] {nombre} registrado. F1-Score: {f1:.4f} | Accuracy: {acc:.4f}")

if __name__ == "__main__":
    entrenar_y_registrar()