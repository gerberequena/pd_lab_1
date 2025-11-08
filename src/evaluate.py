import joblib
import json
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error 
import yaml
import pathlib
import pandas as pd

def read_yaml(file_path):
    try:
        with open(file_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f'Error: the file {file_path} was not found')
        return None

def main():
    #1. cargar yaml file y definir configuraciones
    config = read_yaml('params.yml')

    if config is None:
        return

    file_model = config['evaluate']['file_model']
    metrics_file = config['evaluate']['file_metrics']

    #Cargar modelo y test sets
    data = joblib.load(file_model)
    model = data['model']
    X_test = data['X_test']
    y_test = data['y_test']

    #realizar predicciones
    y_pred = model.predict(X_test)

    # calcular metricas 
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    metrics ={
        'rmse': rmse,
        'mse': mse,
        'mae': mae
    }

    with open(metrics_file, 'w') as file:
        json.dump(metrics, file, indent=4)

    pred_path = config["evaluate"]["out_plot"]
    pathlib.Path(pred_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"y": y_test, "y_pred": y_pred}).to_csv(pred_path, index=False)
    

if __name__ == '__main__':
    main()