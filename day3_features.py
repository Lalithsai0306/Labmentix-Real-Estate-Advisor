import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 1. Load the cleaned dataset
print("Loading cleaned dataset...")
df = pd.read_csv('cleaned_india_housing_prices.csv')

# 2. FEATURE ENGINEERING: Amenities Count
# Buyers care about the NUMBER of amenities. Let's count them!
df['Amenities_Count'] = df['Amenities'].astype(str).apply(lambda x: len(x.split(',')) if x != 'nan' else 0)

# 3. CREATE TARGET LABEL: "Good Investment"
# Business Logic: A property is a "Good Investment" (1) IF:
# - Its Price_per_SqFt is cheaper than the City's average (Undervalued)
# - AND it is relatively new (Age <= 15 years)
# - AND it has High or Medium Public Transport Accessibility
print("\nCalculating Investment Potential...")
city_medians = df.groupby('City')['Price_per_SqFt'].transform('median')

df['Good_Investment'] = (
    (df['Price_per_SqFt'] < city_medians) & 
    (df['Age_of_Property'] <= 15) & 
    (df['Public_Transport_Accessibility'].isin(['High', 'Medium']))
).astype(int) # Converts True/False to 1/0

good_count = df['Good_Investment'].sum()
print(f"Result: Found {good_count} 'Good Investments' out of {len(df)} properties.")

# 4. ENCODING: Convert Text Categories to Numbers
print("\nEncoding text data to numbers for Machine Learning...")
categorical_cols = ['Property_Type', 'Furnished_Status', 'Public_Transport_Accessibility', 
                    'Parking_Space', 'Security', 'Facing', 'Owner_Type', 'Availability_Status', 
                    'State', 'City', 'Locality']

le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# 5. CLEANUP: Drop columns we no longer need
# We drop 'Amenities' because we made 'Amenities_Count', and 'ID' because it's just a row number
df = df.drop(columns=['ID', 'Amenities', 'Year_Built'], errors='ignore')

# 6. Save the Machine Learning-ready dataset!
output_file = 'ml_ready_housing_data.csv'
df.to_csv(output_file, index=False)

print(f"\nDay 3 Complete! ML-Ready data saved to: {output_file}")
print(f"Final Data Shape: {df.shape}")