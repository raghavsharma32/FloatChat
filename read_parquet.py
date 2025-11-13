import pandas as pd
from logger.custom_logger import CustomLogger
from exception.custom_exception import FloatChatException

def read_parquet_file(filepath: str):
    logger = CustomLogger().get_logger("test_parquet")
    try:
        df = pd.read_parquet(filepath)

        logger.info("✅ Parquet file loaded successfully",
                    file=filepath,
                    rows=len(df),
                    columns=list(df.columns))

        print("\n--- Parquet Data Preview ---")
        print(df.head())   # first 5 rows
        print("\n--- Columns ---")
        print(df.columns.tolist())
        print("\n--- Shape ---")
        print(df.shape)

        return df

    except Exception as e:
        raise FloatChatException("Failed to read Parquet file", e)


if __name__ == "__main__":
    # Change to your saved parquet path
    parquet_path = "/Users/raghavsharma/Desktop/Float_Chat/testing/data/sample_argo.parquet"

    print(f"🔎 Reading parquet file: {parquet_path}")
    read_parquet_file(parquet_path)
