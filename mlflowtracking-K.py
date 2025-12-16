""" @ IOC - Joan Quintana - 2024 - CE IABD """

from clustersciclistes import load_dataset, clean, extract_true_labels, clustering_kmeans, homogeneity_score, completeness_score, v_measure_score
import sys
import logging
import shutil
import mlflow

from mlflow.tracking import MlflowClient
sys.path.append("..")


if __name__ == "__main__":
    df = load_dataset("data/ciclistes.csv")
    df = clean(df)
    df_features = df.select_dtypes(include='number')

    true_labels = extract_true_labels(df)

    experiment_name = "K sklearn ciclistes"
    mlflow.set_experiment(experiment_name)

    for K in range(2, 9):
        with mlflow.start_run(run_name=f"K={K}"):
            model = clustering_kmeans(df_features, K)
            predicted_labels = model.labels_

            mlflow.log_param("K", K)
            mlflow.log_metric("homogeneity_score", homogeneity_score(
                true_labels, predicted_labels))
            mlflow.log_metric("completeness_score", completeness_score(
                true_labels, predicted_labels))
            mlflow.log_metric("v_measure_score", v_measure_score(
                true_labels, predicted_labels))

    print('s\'han generat els runs')
