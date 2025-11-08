from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib
import yaml
import numpy as np

def read_yaml(file_path):
    try:
        with open(file_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f'Error: the file {file_path} was not found')
        return None

def main():
    config = read_yaml('params.yml')

    if config is None:
        return

    data_path = config['featurize']['data_path']
    features_cols = config['featurize']['keep_cols'] # Features, sin target
    target = config['featurize']['target']
    file_data_featurized = config['featurize']['FILE_DATA_FEATURIZED']
    file_scaler = config['featurize']['FILE_SCALER'] # Ruta del scaler

    df = pd.read_csv(data_path)

    # 1. Separar X (features) y y (target). El target no se escala.
    X = df[features_cols]
    y = df[target]

    # 2. Escalar solo las features (X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    data_featurized = {
        'X_scaled': X_scaled,
        'y': y.values
    }

    # 3. Guardar el Scaler
    joblib.dump(scaler, file_scaler)
    print(f'Scaler entrenado y guardado en: {file_scaler}')

    # 4. Guardar los datos featurizados
    joblib.dump(data_featurized, file_data_featurized)
    print(f'Datos featurizados y guardados en: {file_data_featurized}')


if __name__ == '__main__':
    main()