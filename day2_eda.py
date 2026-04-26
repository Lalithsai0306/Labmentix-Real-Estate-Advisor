import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the CLEANED dataset
print("Loading cleaned dataset...")
df = pd.read_csv('cleaned_india_housing_prices.csv')

# 2. INSIGHT 1: Average Price per SqFt by City
print("\n--- Top 5 Most Expensive Cities (Avg Price per SqFt) ---")
city_prices = df.groupby('City')['Price_per_SqFt'].mean().sort_values(ascending=False).head(5)
print(city_prices)

# 3. INSIGHT 2: Does Parking Space affect Price?
print("\n--- Average Price (in Lakhs) based on Parking Space ---")
parking_impact = df.groupby('Parking_Space')['Price_in_Lakhs'].mean()
print(parking_impact)

# 4. VISUALIZATION 1: Price by Property Type
plt.figure(figsize=(10, 6))
sns.barplot(x='Property_Type', y='Price_in_Lakhs', data=df, errorbar=None, palette='viridis')
plt.title('Average Property Price by Property Type')
plt.ylabel('Price (in Lakhs)')
plt.xlabel('Property Type')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('price_by_property_type.png') # This saves the image to your folder
print("\nSaved chart: price_by_property_type.png")

# 5. VISUALIZATION 2: Correlation Heatmap (What drives price?)
# Select only numerical columns for correlation
numerical_cols = ['Size_in_SqFt', 'Price_in_Lakhs', 'Price_per_SqFt', 
                  'Age_of_Property', 'Nearby_Schools', 'Nearby_Hospitals']

plt.figure(figsize=(8, 6))
correlation_matrix = df[numerical_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
print("Saved chart: correlation_heatmap.png")

print("\nDay 2 EDA Complete! Check your folder for the new images.")