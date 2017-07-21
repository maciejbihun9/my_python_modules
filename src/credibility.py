

"""
Współczynnik FP = FP/N, gdzie N = TN+FP
 Swoistość = 1–współczynnik FP = TN/N
 Czułość = TP/P, gdzie P = TP+FN
 Precyzja = TP/(TP+FP) - trafne predykcje pozytywne w stosunku do sumy trafnych predykcji pozytywnych oraz fałszywych predykcji pozytywnych.
 Trafność = (TP+TN)/(P+N)
 F-score = precyzja×trafność - ilość odpowiedzi prawdziwych


Trafne predykcje pozytywne (TP) Trafne predykcje negatywne (TN)
Fałszywe Fałszywe predykcje pozytywne (FP) Fałszywe predykcje negatywne (FN)
"""

class Credibility:
    """
    Compute correctness machine learning model regression model.
    """

    def __init__(self, log_reg_res: list):
        self.log_reg_res = log_reg_res
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.get_error_mat()

    def get_specificity(self):
        N = self.TN + self.FP
        return self.TN/N

    def get_sensitivity(self):
        P = self.TP + self.FN
        return self.TP / P

    def get_precision(self):
        return self.TP / (self.TP + self.FP)

    def get_accuracy(self):
        P = self.TP + self.FN
        N = self.TN + self.FP
        return (self.TP + self.TN) / (P + N)

    def get_f_score(self):
        return self.get_precision() / self.get_accuracy()

    def get_error_mat(self):
        for item in self.log_reg_res:
            if item[0] >= 0.5 and item[1] == 1:
                self.TP += 1
            elif item[0] < 0.5 and item[1] == 0:
                self.TN += 1
            elif item[0] < 0.5 and item[1] == 1:
                self.FP += 1
            elif item[0] >= 0.5 and item[1] == 0:
                self.FN += 1

