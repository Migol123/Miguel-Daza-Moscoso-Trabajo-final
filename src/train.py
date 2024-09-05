# Código de Entrenamiento - Modelo de Riesgo de Default en un Banco de Corea
############################################################################


import pandas as pd
import pickle
import os
from sklearn.tree import DecisionTreeClassifier


# Cargar la tabla transformada
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('../data/processed', filename)).set_index('ID')
    X_train = df.drop(["Survived"],axis=1)
    y_train = df[["Survived"]]
    print(filename, ' cargado correctamente')
    # Entrenamos el modelo con toda la muestra
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train, y_train)
    acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)

    print('Modelo entrenado')
    # Guardamos el modelo entrenado para usarlo en produccion
    package = '../models/best_model.pkl'
    pickle.dump(decision_tree, open(package, 'wb'))
    print('Modelo exportado correctamente en la carpeta models')


# Entrenamiento completo
def main():
    read_file_csv('train.csv')
    print('Finalizó el entrenamiento del Modelo')


if __name__ == "__main__":
    main()