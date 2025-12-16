"""
@ IOC - CE IABD
"""
import logging
import pickle

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import homogeneity_score, completeness_score, v_measure_score

logging.basicConfig(level=logging.INFO, format='%(message)s')


def load_dataset(path):
    """
    Carrega el dataset de registres dels ciclistes

    arguments:
            path -- dataset

    Returns: dataframe
    """
    df = pd.read_csv(path)

    return df


def eda(df):
    """
    Exploratory Data Analysis del dataframe

    arguments:
            df -- dataframe

    Returns: None
    """
    df.head()
    df.describe()
    df.info()


def clean(df):
    """
    Elimina les columnes que no són necessàries per a l'anàlisi dels clústers

    arguments:
            df -- dataframe

    Returns: dataframe
    """
    df = df.drop(columns=['dorsal'])
    return df


def extract_true_labels(df):
    """
    Guardem les etiquetes dels ciclistes (BEBB, ...)

    arguments:
            df -- dataframe

    Returns: numpy ndarray (true labels)
    """
    return df['tipus'].values


def visualitzar_pairplot(df):
    """
    Genera una imatge combinant entre sí tots els parells d'atributs.
    Serveix per apreciar si es podran trobar clústers.

    arguments:
            df -- dataframe

    Returns: None
    """

    sns.pairplot(df, hue='labels')
    plt.show()


def clustering_kmeans(data, n_clusters=4):
    """
    Crea el model KMeans de sk-learn, amb 4 clusters (estem cercant 4 agrupacions)
    Entrena el model

    arguments:
            data -- les dades: tp i tb

    Returns: model (objecte KMeans)
    """
    kmeans = KMeans(n_clusters, random_state=42)
    kmeans.fit(data)
    return kmeans


def visualitzar_clusters(data, labels):
    """
    Visualitza els clusters en diferents colors. Provem diferents combinacions de parells d'atributs

    arguments:
            data- el dataset sobre el qual hem entrenat
            labels- array d'etiquetes a què pertanyen les dades
    Returns: None
    """
    df_plot = data.copy()
    df_plot['cluster'] = labels
    sns.pairplot(df_plot, hue='cluster')
    plt.show()


def associar_clusters_patrons(tipus, model):
    """
    Associa els clústers (labels 0, 1, 2, 3) als patrons de comportament (BEBB, BEMB, MEBB, MEMB).
    S'han trobat 4 clústers però aquesta associació encara no s'ha fet.

    arguments:
    tipus -- un array de tipus de patrons que volem actualitzar associant els labels
    model -- model KMeans entrenat

    Returns: array de diccionaris amb l'assignació dels tipus als labels
    """
    # proposta de solució

    dicc = {'tp': 0, 'tb': 1}

    logging.info('Centres:')
    for j in range(len(tipus)):
        logging.info('{:d}:\t(tp: {:.1f}\ttb: {:.1f})'.format(j, model.cluster_centers_[
                     j][dicc['tp']], model.cluster_centers_[j][dicc['tb']]))

    # Procés d'assignació
    ind_label_0 = -1
    ind_label_1 = -1
    ind_label_2 = -1
    ind_label_3 = -1

    suma_max = 0
    suma_min = 50000

    for j, center in enumerate(clustering_model.cluster_centers_):
        suma = round(center[dicc['tp']], 1) + round(center[dicc['tb']], 1)
        if suma_max < suma:
            suma_max = suma
            ind_label_3 = j
        if suma_min > suma:
            suma_min = suma
            ind_label_0 = j

    tipus[0].update({'label': ind_label_0})
    tipus[3].update({'label': ind_label_3})

    lst = [0, 1, 2, 3]
    lst.remove(ind_label_0)
    lst.remove(ind_label_3)

    if clustering_model.cluster_centers_[lst[0]][0] < clustering_model.cluster_centers_[lst[1]][0]:
        ind_label_1 = lst[0]
        ind_label_2 = lst[1]
    else:
        ind_label_1 = lst[1]
        ind_label_2 = lst[0]

    tipus[1].update({'label': ind_label_1})
    tipus[2].update({'label': ind_label_2})

    logging.info('\nHem fet l\'associació')
    logging.info('\nTipus i labels:\n%s', tipus)
    return tipus


def generar_informes(df, tipus):
    """
    Generació dels informes a la carpeta informes/. 
    Tenim un dataset de ciclistes i 4 clústers, i generem
    4 fitxers de ciclistes per cadascun dels clústers

    arguments:
            df -- dataframe
            tipus -- objecte que associa els patrons de comportament amb els labels dels clústers

    Returns: None
    """
    for i in tipus:
        nom = i['name']
        label = i['label']
        path_informes = f'./informes/{nom}.txt'

        df_clust = df[df['cluster'] == label]

        with open(path_informes, "w") as f:
            f.write("temps_pujada,temps_baixada,cluster\n")
            for _, fila in df_clust.iterrows():
                f.write(",".join(str(v) for v in fila.values) + "\n")
    logging.info('S\'han generat els informes en la carpeta informes/\n')


def nova_prediccio(dades, model):
    """
    Passem nous valors de ciclistes, per tal d'assignar aquests valors a un dels 4 clústers

    arguments:
            dades -- llista de llistes, que segueix l'estructura 'id', 'tp', 'tb', 'tt'
            model -- clustering model
    Returns: (dades agrupades, prediccions del model)
    """


# ----------------------------------------------


if __name__ == "__main__":

    PATH_DS = './data/ciclistes.csv'

    dataframe = load_dataset(PATH_DS)
    eda(dataframe)
    dataframe = clean(dataframe)
    true_labels = extract_true_labels(dataframe)
    dataframe = dataframe.drop(columns=['tipus'])
    dataframe['labels'] = true_labels

    visualitzar_pairplot(dataframe)

    dataframe_data = dataframe[['temps_pujada', 'temps_baixada']]
    clustering_model = clustering_kmeans(dataframe_data)
    with open('model/clustering_model.pkl', 'wb') as x:
        pickle.dump(clustering_model, x)
    pred_labels = clustering_model.labels_
    dataframe['cluster'] = pred_labels

    scores = {
        'homogeneity': homogeneity_score(true_labels, pred_labels),
        'completeness': completeness_score(true_labels, pred_labels),
        'v_measure': v_measure_score(true_labels, pred_labels)
    }
    with open('model/scores.pkl', 'wb') as y:
        pickle.dump(scores, y)

    visualitzar_clusters(dataframe_data, pred_labels)

    # array de diccionaris que assignarà els tipus als labels
    tipus_ciclista = [{'name': 'BEBB'}, {'name': 'BEMB'},
                      {'name': 'MEBB'}, {'name': 'MEMB'}]
    tipus_ciclista = associar_clusters_patrons(
        tipus_ciclista, clustering_model)

    tipus_dict = {t['label']: t['name'] for t in tipus_ciclista}
    with open('model/tipus_dict.pkl', 'wb') as z:
        pickle.dump(tipus_dict, z)
    generar_informes(dataframe, tipus_ciclista)

    # Classificació de nous valors
    nous_ciclistes = [
        [500, 3230, 1430, 4670],  # BEBB
        [501, 3300, 2120, 5420],  # BEMB
        [502, 4010, 1510, 5520],  # MEBB
        [503, 4350, 2200, 6550]  # MEMB
    ]

"""Assignació dels nous valors als tipus
    for i, p in enumerate(pred):
        t = [t for t in tipus if t['label'] == p]
        logging.info('tipus %s (%s) - classe %s',
                     df_nous_ciclistes.index[i], t[0]['name'], p)
"""

nous_ciclistes = [
    [500, 3230, 1430, 4660],  # BEBB
    [501, 3300, 2120, 5420],  # BEMB
    [502, 4010, 1510, 5520],  # MEBB
    [503, 4350, 2200, 6550]   # MEMB
]

df_nous = pd.DataFrame(nous_ciclistes, columns=[
                       'id', 'temps_pujada', 'temps_baixada', 'total'])
X_nous = df_nous[['temps_pujada', 'temps_baixada']]

with open('model/clustering_model.pkl', 'rb') as f:
    clustering_model = pickle.load(f)

pred = clustering_model.predict(X_nous)
with open('model/tipus_dict.pkl', 'rb') as f:
    tipus_dict = pickle.load(f)

for i, p in enumerate(pred):
    print(
        f'Ciclista {df_nous.loc[i, "id"]}: Cluster {p} → Tipus {tipus_dict[p]}')
