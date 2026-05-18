import pandas as pd

class DataFeed:
    def __init__(self, file_path: str):
        """
        Loads the dataset from a parquet file.
        Expects a DataFrame with timestamps as the index.
        """
        try:
            # Try reading with pyarrow engine first, fallback to fastparquet if needed
            try:
                self.data = pd.read_parquet(file_path, engine='pyarrow')
            except Exception:
                self.data = pd.read_parquet(file_path, engine='fastparquet')

            # Convert timezone-aware index to naive (remove timezone) if necessary
            if self.data.index.tz is not None:
                self.data.index = self.data.index.tz_localize(None)

        except Exception as e:
            raise IOError(f"Failed to load data from {file_path}. Error: {e}")

        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise ValueError("The index of the dataset must be a DatetimeIndex.")

        if isinstance(self.data.columns, pd.MultiIndex):
            fields = self.data.columns.get_level_values(-1)
            if "close" not in fields:
                raise ValueError("MultiIndex dataset must include a 'close' field.")

        if self.data.isna().any().any():
            raise ValueError("Dataset contains NaN values. Please clean the data before backtesting.")

    def __iter__(self):
        """
        Allows the engine to iterate over the data row-by-row.
        Yields a tuple of (timestamp, pd.DataFrame of market data).
        Output DataFrame format at each timestamp:
        - Index: ticker
        - Columns: fields (e.g. close, volume)
        """
        for timestamp, row in self.data.iterrows():
            if isinstance(self.data.columns, pd.MultiIndex):
                # row index: MultiIndex (ticker, field) -> reshape to DataFrame
                current_market_data = row.unstack(level=-1)
            else:
                # Backward compatibility for close-only datasets
                current_market_data = pd.DataFrame({"close": row})

            yield timestamp, current_market_data