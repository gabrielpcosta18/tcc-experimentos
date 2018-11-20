from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd 
import numpy as np

#Nice graphing tools
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import plotly
import plotly.offline as py
import plotly.tools as tls
import plotly.graph_objs as go
import plotly.tools as tls

from sklearn.metrics import mean_squared_error

def get_rolling_window(timeSeries, window_size):
    arr = [timeSeries.shift(x)[::-1][:window_size].T.values[0].T for x in range(len(timeSeries))[::-1]]
    return pd.DataFrame(arr)

def get_most_recent_window(timeSeries, window_size):
    x = 0
    arr = [timeSeries.shift(x)[::-1][:window_size].T.values[0].T]
    return pd.DataFrame(arr)

def predict_with_knn(ts, column, w, k, train_size, debug=False, cluster_label=None, real_value=True, z_normalization=True):
    predictions = []
    timeSerie = ts[[column]]
    begin_index = int(len(timeSerie)*train_size)

    X, Y = (timeSerie.iloc[:begin_index + 1, : ], timeSerie.iloc[begin_index + 1:, :],)
    X_to_return = timeSerie.iloc[:begin_index + 1, : ]
    # Construir o conjunto de treinamentos S a partir da série X
    # e o tamanho da janela w

    # Normalizando
    normalization = MinMaxScaler()

    if z_normalization:
        # Normalizando com Z-Score
        normalization = StandardScaler()

    X_norm = pd.DataFrame(normalization.fit_transform(X), index=X.index, columns=X.columns)
    S = get_rolling_window(X_norm, w)
    S_backtracking = get_rolling_window(X, w)

    for i in Y.index:
        # Definir a sequência de referência U para qual o valor futuro xn + 1
        # não é conhecido
        U = S.iloc[-1]

        # Obtenção das k sequências mais próximas a U, contidas em S,
        # considerando a medida de similaridade Ms e o critério de seleção
        # de vizinhos próximos Ck
        distances = (S[w - 1:-2] - U).pow(2).sum(1).pow(0.5)                

        # Critério de seleção baseado apenas nos menores
        # base_idx = distances.nsmallest(k).index

        # Critério de seleção baseado nos menores podando pelo tamanho da janela w
        base_idx = pd.Series()
        for x in distances.sort_values(ascending=True).index:
            value = pd.Series(x - base_idx).abs().min()
            if value > w or len(base_idx) == 0:
                base_idx = base_idx.append(pd.Series([x]))
            if len(base_idx) == k:
                break

        class_idx = pd.Series(base_idx + 1)
        # Cálculo do valor futuro da sequência de referência utilizando f(S_)
        value = 0
        norm_value = 0
        for x in class_idx:
            value += S_backtracking.iloc[x][0]
            norm_value += S.iloc[x][0]-S.iloc[x][1]
        
        predictions.append(S.iloc[-1][0] + norm_value/k)

        if debug:
            plt.figure(figsize=(40,16,))
            plt.title(str(column))
            plt.plot(X_norm, linestyle='dashdot', marker='o')
            for idx in base_idx.values:
                plt.plot(X_norm.iloc[idx - w + 1: idx + 1], color='black', marker='o')
            plt.plot(pd.Series(predictions[-1], index=[i]).T, color='red', marker='o')
            plt.show()

        if real_value:
            # Adicionando o valor real
            X.loc[i] = Y.loc[i][0]        
        else:
            # Adicionando o valor previsto
            X.loc[i] = predictions[-1]

        X_to_return.loc[i] = Y.loc[i][0]
        X_norm.loc[i] = normalization.transform([X.loc[i]])[0]
        S.loc[len(S)] = get_most_recent_window(X_norm, w).T.iloc[:, 0]
        S_backtracking.loc[len(S_backtracking)] = get_most_recent_window(X, w).T.iloc[:, 0]
    
    predicted = pd.DataFrame(predictions, index=Y.index, columns=[column])
    return (pd.DataFrame(normalization.transform(X_to_return), index=X_to_return.index, columns=X_to_return.columns), predicted, (predicted - pd.DataFrame(normalization.transform(Y), index=Y.index, columns=Y.columns)).abs().mean(), w, k, cluster_label)

def predict_with_cluster_knn(ts, column, cluster_labels, w, k, train_size, debug=False, real_value=True, z_normalization=True):
    predictions = []
    timeSerie = ts[[column]]
    begin_index = int(len(timeSerie)*train_size)

    X, Y = (timeSerie.iloc[:begin_index + 1, : ], timeSerie.iloc[begin_index + 1:, :],)
    X_to_return = timeSerie.iloc[:begin_index + 1, : ]
    # Construir o conjunto de treinamentos S a partir da série X
    # e o tamanho da janela w
    
    cluster_label = cluster_labels.iloc[ts.T.index.get_loc(column)][0]
    similar_index = cluster_labels[cluster_labels.label==cluster_label].index.values
    similars = ts.T.iloc[similar_index].T
    
    S_similars = pd.DataFrame()
    S_similars_norm = pd.DataFrame()
    for similar in similars:
        current = similars[[similar]].iloc[:begin_index + 1, : ]
        current_norm = pd.DataFrame(StandardScaler().fit_transform(current), index=current.index, columns=current.columns)
        S_similars_norm = S_similars_norm.append(get_rolling_window(current_norm, w))
        empty = pd.Series([1000000000]*w)
        S_similars = S_similars.append(get_rolling_window(current, w))
        empty[0] = S_similars_norm.iloc[-1][0]
        S_similars_norm.iloc[-1] = pd.Series(empty)

    # Normalizando
    normalization = MinMaxScaler()

    if z_normalization:
        # Normalizando com Z-Score
        normalization = StandardScaler()

    X_norm = pd.DataFrame(normalization.fit_transform(X), index=X.index, columns=X.columns)
    S = pd.concat([S_similars_norm, get_rolling_window(X_norm, w)])
    S_backtracking = pd.concat([S_similars, get_rolling_window(X, w)])

    for i in Y.index:
        # Definir a sequência de referência U para qual o valor futuro xn + 1
        # não é conhecido
        U = S.iloc[-1]

        # Obtenção das k sequências mais próximas a U, contidas em S,
        # considerando a medida de similaridade Ms e o critério de seleção
        # de vizinhos próximos Ck
        distances = (S[w - 1:-2] - U).pow(2).sum(1).pow(0.5)                

        # Critério de seleção baseado apenas nos menores
        # base_idx = distances.nsmallest(k).index

        # Critério de seleção baseado nos menores podando pelo tamanho da janela w
        base_idx = pd.Series()
        for x in distances.sort_values(ascending=True).index:
            value = pd.Series(x - base_idx).abs().min()
            if value > w or len(base_idx) == 0:
                base_idx = base_idx.append(pd.Series([x]))
            if len(base_idx) == k:
                break

        class_idx = pd.Series(base_idx + 1)
        # Cálculo do valor futuro da sequência de referência utilizando f(S_)
        value = 0
        norm_value = 0
        for x in class_idx:
            value += S_backtracking.iloc[x][0]
            norm_value += S.iloc[x][0]-S.iloc[x][1]

        predictions.append(S.iloc[-1][0] + norm_value/k)

        if debug:
            plt.figure(figsize=(40,16,))
            plt.title(str(column))
            plt.plot(X, linestyle='dashdot', marker='o')
            for idx in base_idx.values:
                plt.plot(X.iloc[idx - w + 1: idx + 1], color='black', marker='o')
            plt.plot(pd.Series(predictions[-1], index=[i]).T, color='red', marker='o')
            plt.show()

        if real_value:
            # Adicionando o valor real
            X.loc[i] = (Y.loc[i])[0]
        else:
            # Adicionando o valor previsto
            X.loc[i] = predictions[-1]

        X_to_return.loc[i] = Y.loc[i][0]
        X_norm.loc[i] = normalization.transform([X.loc[i]])[0]
        S.loc[len(S)] = get_most_recent_window(X_norm, w).T.iloc[:, 0]
        S_backtracking.loc[len(S_backtracking)] = get_most_recent_window(X, w).T.iloc[:, 0]
    
    predicted = pd.DataFrame(predictions, index=Y.index, columns=[column])
    return (pd.DataFrame(normalization.transform(X_to_return), index=X_to_return.index, columns=X_to_return.columns), predicted, (predicted - pd.DataFrame(normalization.transform(Y), index=Y.index, columns=Y.columns)).abs().mean(), w, k, cluster_label)
