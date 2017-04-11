from math import log
import numpy as np


class model():
    def predict(self, data):
        y = []
        for x in range(len(data.get_data("P"))):
            flux = data.get_data("fluence n/cm2")[x][0]
            tempc = data.get_data("Temp (C)")[x][0]
            p = data.get_data("P")[x][0]
            ni = data.get_data("Ni")[x][0]
            mn = data.get_data("Mn")[x][0]
            cu = data.get_data("Cu")[x][0]
            prod_ID = data.get_data("Product ID")[x][0]

            if prod_ID == 1 or prod_ID == 3 or prod_ID == 0:
                A = 1.080
                B = .819
            elif prod_ID == 2:
                A = .919
                B = .968
            else:
                A = 1.011
                B = .738

            M = B * (max(min(113.87 * (log(flux) - log(4.5e16)), 612.6), 0)) * (((1.8 * tempc + 32) / 550) ** -5.45) * (
                (.1 + p / .012) ** -.098) * ((.168 + ((ni ** .58) / .63)) ** .73)

            tts1 = A * (5 / 9) * 3.593e-10 * (flux ** .5695) * (((1.8 * tempc + 32) / 550) ** -5.47) * (
                (.09 + p / .012) ** .216) * ((1.66 + ((ni ** 8.54) / .63)) ** .39) * ((mn / 1.36) ** .3)
            tts2 = (5 / 9) * max(min(cu, .28) - .053, 0) * M

            tts = tts1 + tts2

            if prod_ID == 1 or prod_ID == 3 or prod_ID == 0:
                cc = .55 + (1.2e-3) * tts - 1.33e-6 * (tts ** 2)
            else:
                cc = .45 + (1.945e-3) * tts - 5.496e-6 * (tts ** 2) + 8.473e-9 * (tts ** 3)

            ds = tts / cc

            y.append(ds)

        return y


def fit(self, x_train, y_train):
    return


def get():
    return model()
