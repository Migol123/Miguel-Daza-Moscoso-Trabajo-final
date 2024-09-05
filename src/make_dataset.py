
# Script de Preparación de Datos
###################################

import pandas as pd
import numpy as np
import os


# Leemos los archivos csv
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('../data/raw/', filename))
    print(filename, ' cargado correctamente')
    return df


# Realizamos la transformación de datos
def data_preparation(df):
    df=df.drop(['Ticket', 'Cabin'], axis=1)
    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    df['Title'] = df['Title'].map(title_mapping)
    df['Title'] = df['Title'].fillna(0)
    df=df.drop(['Name', 'PassengerId'], axis=1)
    df['Sex'] = df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
    df.loc[ df['Age'] <= 16, 'Age'] = 0
    df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1
    df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2
    df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 3
    df.loc[ df['Age'] > 64, 'Age']
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
    df=df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
    df['Age*Class'] = df.Age * df.Pclass
    freq_port = df.Embarked.dropna().mode()[0]
    df['Embarked'] = df['Embarked'].fillna(freq_port)
    df['Embarked'] = df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    df['Fare'].fillna(df['Fare'].dropna().median(), inplace=True)
    df.loc[ df['Fare'] <= 7.91, 'Fare'] = 0
    df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1
    df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare']   = 2
    df.loc[ df['Fare'] > 31, 'Fare'] = 3
    df['Fare'] = df['Fare'].astype(int)
    return df


# Exportamos la matriz de datos con las columnas seleccionadas
def data_exporting(df, features, filename):
    dfp = df[features]
    dfp.to_csv(os.path.join('../data/processed/', filename))
    print(filename, 'exportado correctamente en la carpeta processed')


# Generamos las matrices de datos que se necesitan para la implementación

def main():
    # Matriz de Entrenamiento
    df1 = read_file_csv('train.csv')
    tdf1 = data_preparation(df1)
    data_exporting(tdf1, ['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title','IsAlone', 'Age*Class'],'credit_train.csv')
    # Matriz de Validación
    df2 = read_file_csv('val.csv')
    tdf2 = data_preparation(df2)
    data_exporting(tdf2,['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title','IsAlone', 'Age*Class'],'credit_val.csv')
    # Matriz de Scoring
    df3 = read_file_csv('train.csv')
    tdf3 = data_preparation(df3)
    data_exporting(tdf3,['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title','IsAlone', 'Age*Class'],'credit_score.csv')
    
if __name__ == "__main__":
    main()