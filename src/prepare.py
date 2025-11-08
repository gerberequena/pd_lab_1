import yaml
import pandas as pd


def read_yaml(file_path):
    try:
        with open(file_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f'Error: the file {file_path} was not found')
        return None

def main(drop_na=True):
    config = read_yaml('params.yml')
    
    if config is None:
        return

    target = config['prepare']['target']
    keep_cols = config['prepare']['keep_cols'] # Contiene solo features
    path = config['prepare']['data_path']
    file_date_prep = config['prepare']['FILE_DATA_PREP'] # Corregido a FILE_DATA_PREP


    df = pd.read_csv(path)
    
    if drop_na:
        df = df.dropna()
    
    # Selecciona solo las features definidas y el target
    dataset = df[keep_cols + [target]] 
    dataset.to_csv(file_date_prep, index=False)
    print(f'Datos preparados y guardados en: {file_date_prep}')

if __name__ == '__main__':
    main()