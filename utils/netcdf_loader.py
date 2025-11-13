import os
import xarray as xr
import pandas as pd

from exception.custom_exception import FloatChatException
from logger.custom_logger import CustomLogger


class NetCDFLoader:
    def __init__(self, output_dir: str = "processed_data", qc_filter: bool = True):
        self.logger = CustomLogger().get_logger(__file__)
        self.output_dir = os.path.join(os.getcwd(), output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        self.qc_filter = qc_filter

        # Aliases for ARGO NetCDF fields
        self.col_map = {
            "float_id": ["PLATFORM_NUMBER", "platform_number"],
            "cycle_number": ["CYCLE_NUMBER", "cycle_number"],
            "time": ["JULD", "TIME", "juld"],
            "lat": ["LATITUDE", "latitude"],
            "lon": ["LONGITUDE", "longitude"],
            # prefer adjusted first
            "pressure_dbar": ["PRES_ADJUSTED", "PRES", "pres_adjusted", "pres"],
            "temperature": ["TEMP_ADJUSTED", "TEMP", "temp_adjusted", "temp"],
            "salinity": ["PSAL_ADJUSTED", "PSAL", "psal_adjusted", "psal"],
            "qc_temp": ["TEMP_QC", "temp_qc"],
            "qc_salin": ["PSAL_QC", "psal_qc"],
            "qc_pres": ["PRES_QC", "pres_qc"],
        }

    def load_netcdf(self, filepath: str) -> pd.DataFrame:
        """
        Load ARGO NetCDF file into a pandas DataFrame.
        Extract essential columns: float_id, cycle_number, time, lat, lon,
        pressure (dbar), temperature, salinity, QC flags.
        Applies QC filtering if enabled.
        """
        try:
            if not os.path.exists(filepath):
                raise FloatChatException(f"NetCDF file not found: {filepath}")

            # Load dataset
            ds = xr.open_dataset(filepath)
            df = ds.to_dataframe().reset_index()

            # Map aliases → standard names
            rename_dict = {}
            for target, aliases in self.col_map.items():
                for candidate in aliases:
                    if candidate in df.columns:
                        rename_dict[candidate] = target
                        break  # stop at first available match

            if not rename_dict:
                self.logger.warning("⚠️ No matching columns found", file=filepath)
                return pd.DataFrame()

            # Keep only mapped columns
            df = df[list(rename_dict.keys())].rename(columns=rename_dict)

            # ---- Normalize fields ----
            if "float_id" in df.columns:
                df["float_id"] = df["float_id"].astype(str).str.strip()

            if "cycle_number" in df.columns:
                df["cycle_number"] = pd.to_numeric(df["cycle_number"], errors="coerce").astype("Int64")

            # Convert ARGO JULD (days since 1950-01-01) → datetime
            if "time" in df.columns and pd.api.types.is_numeric_dtype(df["time"]):
                df["time"] = pd.to_datetime(
                    df["time"], origin="1950-01-01", unit="D", errors="coerce"
                )

            # ---- QC Filtering ----
            if self.qc_filter:
                qc_cols = {"temperature": "qc_temp", "salinity": "qc_salin", "pressure_dbar": "qc_pres"}
                for var, qc in qc_cols.items():
                    if var in df.columns and qc in df.columns:
                        mask = df[qc].astype(str).isin(["1", "2"])  # good or probably good
                        df = df[mask | df[qc].isna()]  # keep rows where QC not provided

            # ---- Add depth ----
            if "pressure_dbar" in df.columns:
                df["depth"] = pd.to_numeric(df["pressure_dbar"], errors="coerce")
            else:
                df["depth"] = pd.NA  # ensure column exists even if missing

            # Drop rows with missing essential fields
            df = df.dropna(subset=["lat", "lon", "depth"]).drop_duplicates()

            self.logger.info(
                "✅ NetCDF loaded and filtered",
                file=filepath,
                rows=len(df),
                columns=list(df.columns),
                mapped_columns=rename_dict,
            )

            return df

        except Exception as e:
            raise FloatChatException("Failed to load NetCDF file", e) from e

    def save(self, df: pd.DataFrame, base_filename: str):
        """
        Save dataframe to both CSV and Parquet in processed_data/ folder.
        """
        try:
            if df.empty:
                self.logger.warning("⚠️ Attempted to save empty DataFrame")
                return None, None

            csv_path = os.path.join(self.output_dir, f"{base_filename}.csv")
            parquet_path = os.path.join(self.output_dir, f"{base_filename}.parquet")

            df.to_csv(csv_path, index=False)
            df.to_parquet(parquet_path, index=False)

            self.logger.info(
                "✅ Data saved",
                csv_file=csv_path,
                parquet_file=parquet_path,
                rows=len(df),
                columns=list(df.columns),
            )

            return csv_path, parquet_path

        except Exception as e:
            raise FloatChatException("Failed to save DataFrame", e) from e
