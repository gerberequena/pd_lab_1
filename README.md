1.  Primero creamos git init y luego dvc init para iniciar los repositorios correspondientes

2.  Ya que que se trabaja localmente usamos el comando: dvc remote add -d localstore C:/dvcstore, ya que si se descarga desde el repositorio, se necesita
    pegar el dataset_v1.csv en la carpeta data y anadirlor por medio de dvc add data/dataset_v1.csv y git add data/dataset_v1.csv.dvc y data/.gitignore

3.  Creacion del pipeline que hizo en 4 pasos para mejor manejo de stages
    3.1 prepare: Es un script que ayuda a limpiar el dataset y toma en cuenta unicamente las columnas que se dan en params.yml
    3.2 feafurizd: Es un script que aplica Standar Scaler y se podrian anadir transformacion de variables si fuera necesario.
    3.3 train: entrena un modelo dado en este caso usamos linearregressor y tambien probamos SGDRegressor y se pasan hiperparametros por medio de params.yml en caso de usar SGDRegressor
    3.4 Se crea evaluaaciones para medir las metricas como mean_squared_error, root_mean_squared_error, mean_absolute_error se puede usar dvc plots show plots/pred_vs_actual.csv para mostrar graficos

4.  En este caso se creo un pipeline reproducible con ajustes que se pueden hacer en params.yml o en los scripts en dado caso se quisieran anadir otro tipo de modelos
    dentro de la experimetnacion se hizo por medio de LinearRegressor y SGDRegressor el cual Linear regressor mostros mejores resultaos, ya que se selecciono ese modelo
    decidimos simular un cambio de dataset por medio de data_exploration.ipynb en el cual removimos outlayers sobre MedianHouseVal, con lo cual el modelo mejoro su rendimiento
    en comparacion del primero, se puede hacer una comparacion de metricas por medio de dvc exp list y luego tomar los tags y usar dvc exp diff tag1 tag2, por lo cual nos quedamos
    con el ultimo modelo.
