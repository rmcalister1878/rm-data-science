import matplotlib.pyplot as plt
from scipy.stats import shapiro, pearsonr, spearmanr
from scipy.stats import pointbiserialr
import statsmodels.api as sm
from statsmodels.formula.api import ols

from visualisation.plotting_helpers import grid_plots


class Correlation(object):
    def __init__(self, target_column, categorical_columns=None, numerical_columns=None,
                 binary_categorical_columns=None):
        self.target_column = target_column
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.binary_categorical_columns = binary_categorical_columns


    def visualise_numerical_correlation(self, df):
        if self.numerical_columns is not None:
            axes = grid_plots(self.numerical_columns)
            for i, col in enumerate(self.numerical_columns):
                axes[i].scatter(df[col], df[self.target_column])
                axes[i].set_xlabel(col)
                axes[i].set_ylabel(self.target_column)


    def determine_normality(self, df):
        column_normality = {}
        if self.numerical_columns is not None:
            axes = grid_plots(self.numerical_columns)
            for i, col in enumerate(self.numerical_columns):
                stat, p_value = shapiro(df[col])
                axes[i].hist(df[col], bins=15, edgecolor='black')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
                print(f"{col} Shapiro-Wilk Test: Statistic = {stat}, p-value = {p_value}")
                # If p-value > 0.05, data is likely normal
                if p_value > 0.05:
                    print("The data follows a normal distribution.")
                    column_normality[col] = True
                else:
                    print("The data does not follow a normal distribution.")
                    column_normality[col] = False
        return column_normality


    def calculate_correlation_coefficients(self, df, column_normality):
        for column in self.numerical_columns:
            if column_normality[column]:
                stat, p_val = pearsonr(df[column], df[self.target_column])
            else:
                stat, p_val = spearmanr(df[column], df[self.target_column])
            print(f"{column} - Correlation Coefficient = {stat}, p-value = {p_val}")
        for column in self.binary_categorical_columns:
            corr, p_value = pointbiserialr(df[column], df[self.target_column])
            print(f"{column} - Point Biserial Correlation Coefficient = {corr}, p-value = {p_value}")
        for column in self.categorical_columns:
            model = ols(f"{self.target_column} ~ C({column})", data=df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            f_stat = anova_table['F'][0]
            p_value = anova_table['PR(>F)'][0]
            print(f"{column} - Anova F = {f_stat}, p-value = {p_value}")


