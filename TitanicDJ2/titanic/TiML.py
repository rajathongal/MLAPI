import numpy as np
import pandas as pd
import os
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle

np.random.seed(42)
warnings.filterwarnings(action="ignore", message="^internal gelsd")


def training(train_raw):
    train = train_raw.drop(["PassengerId", "Name", "Ticket", "Cabin", "Fare"], axis=1)
    X = train.drop("Survived", axis=1)
    y = train["Survived"].copy()
    X_prepared = pre_proc(X)
    dummyRow = pd.DataFrame(np.zeros(len(X.columns)).reshape(1, len(X.columns)), columns=X.columns)
    dummyRow.to_csv("dummyRow.csv", index=False)
    model = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_prepared, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    pkl_filename = "pickle_model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)
    y_pred = model.predict(X_test)
    print(model.score(X_prepared, y))
    print("sur", sum(y_pred != 0))
    print("not sur", sum(y_pred == 0))
    cnf_mat = confusion_matrix(y_test, y_pred)
    print(cnf_mat)


def pre_proc(X):
    class MostFrequentImputer(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X], index=X.columns)
            return self

        def transform(self, X, y=None):
            return X.fillna(self.most_frequent_)

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", MostFrequentImputer()),
        ("cat_encoder", OneHotEncoder(sparse=False)),
    ])

    num_attribs = ["Age", "SibSp", "Parch"]
    cat_attribs = ["Pclass", "Sex", "Embarked"]

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs),
    ])

    X_prepared = full_pipeline.fit_transform(X)
    return X_prepared


def pred(ob):
    d1 = ob.to_dict()
    df = pd.DataFrame(d1, index=[0])
    df.drop("Survived", axis="columns", inplace=True)
    df = pre_proc(df)
    dummyrow_filename = "dummyRow.csv"
    dummyrow_filename = os.path.dirname(__file__) + "/" + dummyrow_filename
    df2 = pd.read_csv(dummyrow_filename)
    for c1 in df.columns:
        df2[c1] = df[c1]
    pkl_filename = "pickle_model.pkl"
    pkl_filename = os.path.dirname(__file__) + "/" + pkl_filename
    with open(pkl_filename, 'rb') as file:
        model = pickle.load(file)
    pred = model.predict(df2)
    return pred


if __name__ == "__main__":
    data_Path = os.path.join("datasets", "Classification")


    def load_data(file_name, data_path=data_Path):
        csv_path = os.path.join(data_path, file_name)
        return pd.read_csv(csv_path)


    train_raw = load_data("titanic_train.csv")
    training(train_raw)
