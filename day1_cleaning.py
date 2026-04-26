import pandas as pd

# 1. Load the dataset
df = pd.read_csv('india_housing_prices.csv')

# 2. Structural Integrity: Standardize Categorical Strings
df['City'] = df['City'].astype(str).str.lower().str.strip()
df['Locality'] = df['Locality'].astype(str).str.lower().str.strip()
df['State'] = df['State'].astype(str).str.title().str.strip()
df['Property_Type'] = df['Property_Type'].astype(str).str.title().str.strip()

# 3. Create a clean base feature: Age of Property
current_year = 2026
df['Age_of_Property'] = current_year - df['Year_Built']

# 4. Save the preprocessed dataset
cleaned_file_path = 'cleaned_india_housing_prices.csv'
df.to_csv(cleaned_file_path, index=False)

print(f"Dataset successfully cleaned and standardized. Saved to: {cleaned_file_path}")
print(f"Total Rows: {df.shape[0]}, Total Columns: {df.shape[1]}")