import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import yaml

def read_yaml(file_path):
    try:
        with open(file_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f'Error: the file {file_path} was not found')
        return None

def main():
    #1. cargar yaml file y configuraciones
    config = read_yaml('params.yml')

    if config is None:
        return
    
    file_data_featurized = config['train']['file_path']
    random_state = config['train']['random_state']
    test_size = config['train']['test_size']
    file_model = config['train']['file_model']
    model_params = config['train']['model_params'] # Parámetros del modelo
    
    #2. cargar featurized dataset 
    data = joblib.load(file_data_featurized)

    #3. split dataset en target(y) y features(X)
    X = data['X_scaled']
    y = data['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    #4. seleccionar modelo y entrenar
    # Inicializar el modelo con los parámetros de configuración
    #model = SGDRegressor(**model_params)
    model = LinearRegression()
    model.fit(X_train, y_train)

    #5. Guardar modelo, X_test y y_test
    joblib.dump({
        'model':model,
        'X_test': X_test,
        'y_test': y_test
    }, file_model)
    print(f'Modelo entrenado y guardado en: {file_model}')

if __name__ == '__main__':
    main()