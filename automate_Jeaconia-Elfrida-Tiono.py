import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from joblib import dump


def preprocess_data(
    WineQT,
    target_column="quality",
    test_size=0.3,
    random_state=42,
    save_pipeline_path="preprocessing_pipeline.joblib"
):
    """
    Fungsi Preprocessing Data WineQT
    """

    # Step 1: Load Data
    df = pd.read_csv(WineQT)

    # Drop kolom Id jika ada
    df = df.drop(columns=["Id"], errors="ignore")

    # Step 2: Data Cleaning
    # Drop duplicate
    df = df.drop_duplicates()

    # Step 3: Pisahkan fitur & target
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Step 4: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Step 5: Imputasi missing value (mean)
    imputer = SimpleImputer(strategy="mean")

    X_train = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )

    X_test = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    # Step 6: Outlier handling (IQR capping)
    for col in X_train.columns:
        Q1 = X_train[col].quantile(0.25)
        Q3 = X_train[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        X_train[col] = X_train[col].clip(lower, upper)
        X_test[col] = X_test[col].clip(lower, upper)

    # Step 7: Feature scaling
    scaler = MinMaxScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    # Step 8: Simpan pipeline (imputer + scaler)
    preprocessing_pipeline = Pipeline(steps=[
        ("imputer", imputer),
        ("scaler", scaler)
    ])

    dump(preprocessing_pipeline, save_pipeline_path)

    return X_train_scaled, X_test_scaled, y_train, y_test

X_train, X_test, y_train, y_test = preprocess_data(
    WineQT="WineQT.csv",
    target_column="quality",
    save_pipeline_path="preprocessing_pipeline.joblib"
)

print("Preprocessing selesai")
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
