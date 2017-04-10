from math import log
import numpy as np


def effectiveFluence(Fl, Fx, Ref_Fx=4.39e10, p=.259):
    effFl = Fl

    if Fx < Ref_Fx:
        effFl = Fl * ((Ref_Fx / Fx) ** p)

    return effFl


def effectiveCu(Cu, isWeld):
    effCu = Cu
    if isWeld:
        MaxCu = .243
    else:
        MaxCu = .301

    if Cu <= .072:
        effCu = 0
    else:
        effCu = np.minimum(Cu, MaxCu)

    return effCu


def f(effCu, P):
    if effCu <= .072:
        f = 0
    elif P < .008:
        f = (effCu - .072) ** .668
    else:
        f = (effCu - .072 + 1.359 * (P - .008)) ** .668

    return f


def g(effCu, Ni, effFl):
    return .5 + .5 * np.tanh((np.log10(effFl) + 1.139 * effCu - .448 * Ni - 18.120) / .629)


def MF(prod_ID, Temp, P, Mn, effFl):
    if prod_ID == 1 or prod_ID == 3:
        A = 1.561e-7
    elif prod_ID == 2:
        A = 1.417e-7
    elif prod_ID == 4:
        A = 1.140e-7
    else:
        A = 1.561e-7

    MF = A * (1 - .001718 * Temp) * (1 + 6.13 * P * (Mn ** 2.47)) * ((effFl) ** .5)

    return MF


def CRP(prod_ID, Ni, effCu, P, effFl):
    if prod_ID == 1:
        B = 102.5
    elif prod_ID == 2:
        B = 155.0
    elif prod_ID == 3:
        B = 128.2
    elif prod_ID == 4:
        B = 102.3
    else:
        B = 102.5

    CRP = B * (1 + 3.77 * (Ni ** 1.191)) * (f(effCu, P)) * g(effCu, Ni, effFl)

    return CRP


class model():
    def fit(self, x_train, y_train):
        return

    def predict(self, data):
        # y = np.zeros(len(data.get_data()[0]))
        y = []
        for x in range(len(data.get_data("P"))):

            prod_ID = data.get_data("Product ID")[x][0]
            fluence = data.get_data("fluence n/cm2")[x][0]
            tempc = data.get_data("Temp (C)")[x][0]
            p = data.get_data("P")[x][0]
            ni = data.get_data("Ni")[x][0]
            mn = data.get_data("Mn")[x][0]
            cu = data.get_data("Cu")[x][0]
            flux = data.get_data("flux n/cm2/s")[x][0]

            Temp = tempc * 1.8 + 32

            effFl = effectiveFluence(fluence, flux)
            effCu = effectiveCu(cu, data.get_data("isWeld")[x][0])

            # divide by .7 to get into ∆sigma, divide by 1.8 to get into C
            tts = (MF(prod_ID, Temp, p, mn, effFl) + CRP(prod_ID, ni, effCu, p, effFl)) / 1.8

            if prod_ID == 1 or prod_ID == 2 or prod_ID == 0:
                cc = .55 + (1.2e-3) * tts - 1.33e-6 * (tts ** 2)
            else:
                cc = .45 + (1.945e-3) * tts - 5.496e-6 * (tts ** 2) + 8.473e-9 * (tts ** 3)

            ds = tts / cc

            y.append(ds)
        return y


def get():
    return model()
