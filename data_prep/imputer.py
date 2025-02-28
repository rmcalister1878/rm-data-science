from sklearn.impute import KNNImputer


class Imputer(object):
    def __init__(self, categorical_columns=None, numerical_columns=None):
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns

    def impute_median(self, df):
        if self.numerical_columns is not None:
            for column in self.numerical_columns:
                if df[column].isnull().values.any():
                    df[column] = df[column].fillna(df[column].median())
        else:
            print("No numerical columns")
        return df

    def impute_mean(self, df):
        if self.numerical_columns is not None:
            for column in self.numerical_columns:
                if df[column].isnull().values.any():
                    df[column] = df[column].fillna(df[column].mean())
        else:
            print("No numerical columns")
        return df

    def impute_knn(self, df):
        if self.numerical_columns is not None:
            imputer = KNNImputer(n_neighbors=5)
            df[self.numerical_columns] = imputer.fit_transform(df[self.numerical_columns])
        else:
            print("No numerical columns")
        return df


    def impute_mode(self, df):
        if self.categorical_columns is not None:
            for column in self.categorical_columns:
                if df[column].isnull().values.any():
                    df[column].fillna(df[column].mode()[0])
        else:
            print("No Categorical columns")
        return df
