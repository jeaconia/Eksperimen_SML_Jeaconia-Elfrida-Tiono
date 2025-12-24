import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from joblib import dump

def preprocess_data(
    dataset_path,
    target_column="Rings",
    test_size=0.3,
    random_state=42
):
    """
    Fungsi Preprocessing Dataset yang mengembalikan data siap latih.
    """
    
    # Step 1: Load Data
    df = pd.read_csv(dataset_path)

    # Drop kolom kategori (Sex) jika tidak dilakukan encoding
    df = df.drop(columns=["Sex"], errors="ignore")

    # Step 2: Data Cleaning
    df = df.drop_duplicates()

    # Step 3: Split fitur & target
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Step 4: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    # Step 5: Imputasi Missing Value
    imputer = SimpleImputer(strategy="mean")

    # Fit dan transform pada data train, transform pada data test
    X_train = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns
    )

    X_test = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns
    )

    # Step 6: Outlier Handling (IQR) - Clipping
    for col in X_train.columns:
        Q1 = X_train[col].quantile(0.25)
        Q3 = X_train[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        X_train[col] = X_train[col].clip(lower, upper)
        X_test[col] = X_test[col].clip(lower, upper)

    # Step 7: Scaling
    scaler = MinMaxScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns
    )

    # Step 8: Save Processed Data (Optional side effect)
    processed_dir = "preprocessing/Abalone_preprocessing"
    model_dir = "models"
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    X_train_scaled.to_csv(f"{processed_dir}/X_train.csv", index=False)
    X_test_scaled.to_csv(f"{processed_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{processed_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{processed_dir}/y_test.csv", index=False)

    # Step 9: Save Pipeline
    pipeline = Pipeline([
        ("imputer", imputer),
        ("scaler", scaler)
    ])
    dump(pipeline, f"{model_dir}/preprocessing_pipeline.joblib")

    print("✅ Preprocessing selesai")
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    # Sekarang kita bisa menangkap hasil return dari fungsi tersebut
    X_train, X_test, y_train, y_test = preprocess_data(
        dataset_path="Abalone_raw/abalone.csv",
        target_column="Rings"
    )
    
    # Cek apakah data berhasil di-return
    print(f"Bentuk X_train: {X_train.shape}")
