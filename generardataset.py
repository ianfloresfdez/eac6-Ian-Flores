import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(message)s')


def generar_dataset(num, ind, dicc):
    """
    Genera els temps dels ciclistes, de forma aleatòria, però en base a la informació del diccionari
    """

    dataset = []
    for i in range(num):

        tipus = dicc

        temps_pujada = int(np.random.normal(
            tipus["mu_p"], tipus["sigma"]))
        temps_baixada = int(np.random.normal(
            tipus["mu_b"], tipus["sigma"]))

        dataset.append({
            "dorsal": ind + i,
            "tipus": tipus["name"],
            "temps_pujada": temps_pujada,
            "temps_baixada": temps_baixada
        })
    return dataset


if __name__ == "__main__":

    STR_CICLISTES = 'data/ciclistes.csv'

    # BEBB: bons escaladors, bons baixadors
    # BEMB: bons escaladors, mal baixadors
    # MEBB: mal escaladors, bons baixadors
    # MEMB: mal escaladors, mal baixadors

    # Port del Cantó (18 Km de pujada, 18 Km de baixada)
    # pujar a 20 Km/h són 54 min = 3240 seg
    # pujar a 14 Km/h són 77 min = 4268 seg
    # baixar a 45 Km/h són 24 min = 1440 seg
    # baixar a 30 Km/h són 36 min = 2160 seg
    MU_P_B = 3240  # mitjana temps pujada bons escaladors
    MU_P_ME = 4268  # mitjana temps pujada mals escaladors
    MU_B_BB = 1440  # mitjana temps baixada bons baixadors
    MU_B_MB = 2160  # mitjana temps baixada mals baixadors
    SIGMA = 240  # 240 s = 4 min

    diccionari = [
        {"name": "BEBB", "mu_p": MU_P_B, "mu_b": MU_B_BB, "sigma": SIGMA},
        {"name": "BEMB", "mu_p": MU_P_B, "mu_b": MU_B_MB, "sigma": SIGMA},
        {"name": "MEBB", "mu_p": MU_P_ME, "mu_b": MU_B_BB, "sigma": SIGMA},
        {"name": "MEMB", "mu_p": MU_P_ME, "mu_b": MU_B_MB, "sigma": SIGMA}
    ]

    ds = generar_dataset(50, 1, diccionari)

    with open(STR_CICLISTES, "w", newline="") as f:
        f.write("dorsal,tipus,temps_pujada,temps_baixada\n")
        for fila in ds:
            f.write(",".join(str(v) for v in fila.values()) + "\n")
    logging.info("s'ha generat data/ciclistes.csv")
