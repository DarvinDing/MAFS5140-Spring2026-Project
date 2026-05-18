import pandas as pd

# Install fastparquet first if not installed: pip install fastparquet
df = pd.read_parquet('validation.parquet', engine='fastparquet')

print("Data loaded successfully with fastparquet!")
print(f"Shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn names and types:")
print(df.dtypes)