from collections import Counter

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, KBinsDiscretizer

from category_encoders.one_hot import OneHotEncoder
from category_encoders.ordinal import OrdinalEncoder

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler



def detect_outliers(df, n, features):
    '''
    Função responsável para detectar outliers no dataset
    
    --------
    paramers:
        df: dataset para analise
        type: DataFrame ou Series
        
        n: informar o valor de até quantos outliers serão capturados
        type: int
        
        features: atributos da base de dados
        type: list
        
        return: list
        
    '''
    
    outlier_indices = []
    
    for col in features:
        Q1 = np.percentile(df[col], 25) # Calcula o primeiro quartil
        Q3 = np.percentile(df[col], 75) # calcula o terceiro quartil
        
        IQR = Q3 - Q1 # Calcula o intervalo intequartil
        
        outlier_step = 1.5 * IQR # Multiplica o intervalo intequartil por 1.5
        
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index # retorna os índices dos registros que contém outlier
        
        outlier_indices.extend(outlier_list_col) # lista que contém os índices
        
    outlier_indices = Counter(outlier_indices) # transforma esses índices e as contagens dos mesmos em um dicionário
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n) # retorna os índices
    
    return multiple_outliers



def detects_unbalanced_classes(df, col):
    ''' 
    Função responsável por detectar se as classes estão desbalanceadas
    --------
    paramers:
    
        df: dataframe para análise
        type: pandas.Dataframe
        
        col: coluna
        type: str
        
        return float
    '''
    classes = list(df[col].value_counts(normalize=True).values)
    return (classes[0] - classes[1]) * 100



def conditional_entropy(X, y):
    """
    Calculates the conditional entropy of x given y: S(x|y)
    Wikipedia: https://en.wikipedia.org/wiki/Conditional_entropy
    :param x: list / NumPy ndarray / Pandas DataFrame
        A sequence of measurements
    :param y: list / NumPy ndarray / Pandas DataFrame
        A sequence of measurements
    :return: float
    """

    # entropy of x given y

    y_counter = Counter(y)

    xy_counter = Counter(list(zip(X, y)))

    total_occurrences = sum(y_counter.values())

    entropy = 0.0

    for xy in xy_counter.keys():

        p_xy = xy_counter[xy] / total_occurrences

        p_y = y_counter[xy[1]] / total_occurrences

        entropy += p_xy * np.log(p_y / p_xy)

    return entropy




def binning(df, n_bins=None, encode=None, strategy=None):
    
    data = df.copy()
    
    if isinstance(data, pd.DataFrame):
        for col in data.columns:
            discretizer = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)
            discretizer.fit(data[col].values.reshape(-1, 1))
            data[col] = discretizer.transform(data[col].values.reshape(-1, 1))
        
        return data
    elif isinstance(data, pd.Series):
        discretizer = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)
        
        col_name = data.name
        discretizer.fit(data.values.reshape(-1, 1))
        data = discretizer.transform(data.values.reshape(-1, 1))
        
        dt = pd.DataFrame(data, columns=[col_name])
        
        return dt


def scaling(df, target_name=None):
    '''
    Aplica normalização nas variáveis numéricas
    
    --------
    paramers:

        data:features para transformação
        type: pandas.DataFrame
        
        target_name: nome da variável alvo
        type: str
        
    return pandas.DataFrame
    '''
    
    data = df.copy()
    
    if isinstance(data, pd.DataFrame):
        for col in data.columns:
            if col not in target_name:
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaler.fit(data[col].values.reshape(-1, 1))
                data[col] = scaler.transform(data[col].values.reshape(-1, 1))
        
        return data
    elif isinstance(data, pd.Series):
        scaler = MinMaxScaler(feature_range=(0, 1))
        
        col_name = data.name
        scaler.fit(data.values.reshape(-1, 1))
        data = scaler.transform(data.values.reshape(-1, 1))
        
        dt = pd.DataFrame(data, columns=[col_name])
        
        return dt
        


def standardization(df, target_name=None):
    '''
    Aplica padronização nas variáveis numéricas
    
    --------
    paramers:

        data:features para transformação
        type: pandas.DataFrame
        
        target_name: nome da variável alvo
        type; str
        
    return pandas.DataFrame
    '''
    
    data = df.copy()
    
    if isinstance(data, pd.DataFrame):
        for col in data.columns:
            if col not in target_name:
                scaler = StandardScaler()
                scaler.fit(data[col].values.reshape(-1, 1))
                data[col] = scaler.transform(data[col].values.reshape(-1, 1))
        
        return data
    elif isinstance(data, pd.Series):
        scaler = StandardScaler()
        
        col_name = data.name
        scaler.fit(data.values.reshape(-1, 1))
        data = scaler.transform(data.values.reshape(-1, 1))
        
        dt = pd.DataFrame(data, columns=[col_name])
        
        return dt


def onehot_encoder(df):
    '''
    Aplica onehot encoder nas variáveis categóricas nominais
    
    --------
    paramers:

        data:features para transformação
        type: pandas.DataFrame
        
    return pandas.DataFrame
    '''
    
    data = df.copy()
    
    onehot = OneHotEncoder()
    onehot.fit(data)
    transform = onehot.transform(data)
    
    return transform



def ordinal_encoder(df):
    '''
    Aplica ordinal encoder nas variáveis categóricas ordinais
    
    --------
    paramers:

        data:features para transformação
        type: pandas.DataFrame
        
    return pandas.DataFrame
    '''
    
    data = df.copy()
    
    ordinal_encoder = OrdinalEncoder()
    ordinal_encoder.fit(data)
    transform = ordinal_encoder.transform(data)
    
    return transform


def over_sampling(X, y):
    '''
    Aplica o método over sampling para balancear as classes
    
    --------
    paramers:

        X: variáveis independentes
        type: numpy.array 2D
        
        y: variável dependente
        type: numpy.array 1D
        
    return pandas.DataFrame
    '''
    
    oversampling = SMOTE()
    X_resampled, y_resampled = oversampling.fit_resample(X, y)
    
    return X_resampled, y_resampled

def under_sampling(X, y):
    '''
    Aplica o método under sampling para balancear as classes
    
    --------
    paramers:

        X: variáveis independentes
        type: numpy.array 2D
        
        y: variável dependente
        type: numpy.array 1D
        
    return pandas.DataFrame
    '''
    
    undersample = RandomUnderSampler()
    X_resampled, y_resampled = undersample.fit_sample(X, y)
    
    return X_resampled, y_resampled