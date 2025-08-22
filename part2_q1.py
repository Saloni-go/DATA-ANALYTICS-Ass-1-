import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('M25_DA_A1_Dataset2.csv')

# Data cleaning and preparation
# Convert price to numeric if it's stored as string
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

# Remove rows with missing prices
df = df.dropna(subset=['Price'])

# Define price ranges (low, medium, high) using percentiles
price_percentiles = df['Price'].quantile([0.33, 0.66]).values
df['Price Range'] = pd.cut(df['Price'], 
                          bins=[0, price_percentiles[0], price_percentiles[1], df['Price'].max()],
                          labels=['Low', 'Medium', 'High'])

# Create a figure with multiple subplots
plt.figure(figsize=(20, 16))

# 1. Distribution of price ranges across cities (stacked bar chart)
plt.subplot(3, 3, 1)
price_city_dist = pd.crosstab(df['City'], df['Price Range'], normalize='index') * 100
price_city_dist.plot(kind='bar', stacked=True, ax=plt.gca(), colormap='viridis')
plt.title('Price Range Distribution Across Cities (%)', fontsize=12, fontweight='bold')
plt.ylabel('Percentage of Properties', fontsize=10)
plt.xlabel('City', fontsize=10)
plt.legend(title='Price Range', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45, ha='right')

# 2. Average price by city
plt.subplot(3, 3, 2)
city_avg_prices = df.groupby('City')['Price'].mean().sort_values(ascending=False)
city_avg_prices.plot(kind='bar', color='skyblue')
plt.title('Average Property Price by City', fontsize=12, fontweight='bold')
plt.ylabel('Average Price', fontsize=10)
plt.xlabel('City', fontsize=10)
plt.xticks(rotation=45, ha='right')
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'₹{x/1000000:.1f}M'))

# 3. Property type distribution across price ranges
plt.subplot(3, 3, 3)
prop_type_dist = pd.crosstab(df['Type of Property'], df['Price Range'], normalize='columns') * 100
prop_type_dist.plot(kind='bar', ax=plt.gca(), colormap='plasma')
plt.title('Property Type Distribution by Price Range (%)', fontsize=12, fontweight='bold')
plt.ylabel('Percentage', fontsize=10)
plt.xlabel('Property Type', fontsize=10)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Price Range', bbox_to_anchor=(1.05, 1), loc='upper left')

# 4. Number of bedrooms distribution by price range
plt.subplot(3, 3, 4)
bedroom_dist = pd.crosstab(df['bedroom'], df['Price Range'], normalize='columns') * 100
bedroom_dist.plot(kind='bar', ax=plt.gca(), colormap='cool')
plt.title('Bedroom Distribution by Price Range (%)', fontsize=12, fontweight='bold')
plt.ylabel('Percentage', fontsize=10)
plt.xlabel('Number of Bedrooms', fontsize=10)
plt.legend(title='Price Range', bbox_to_anchor=(1.05, 1), loc='upper left')

# 5. Amenities heatmap by price range
plt.subplot(3, 3, 5)
amenities_columns = ['Swimming Pool', 'Gymnasium', 'Park', 'Security', 'Lift', 'Club House']
# Filter amenities that exist in the dataset
available_amenities = [col for col in amenities_columns if col in df.columns]

if available_amenities:
    amenity_rates = df.groupby('Price Range')[available_amenities].mean() * 100
    sns.heatmap(amenity_rates, annot=True, fmt='.1f', cmap='YlOrRd', 
                cbar_kws={'label': 'Availability (%)'})
    plt.title('Amenity Availability by Price Range (%)', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45)
else:
    plt.text(0.5, 0.5, 'Amenity data not available', ha='center', va='center')
    plt.title('Amenity Data Not Available', fontsize=12, fontweight='bold')

# 6. Price distribution by city (boxplot)
plt.subplot(3, 3, 6)
city_data = df[df['City'].isin(df['City'].value_counts().head(8).index)]  # Top 8 cities
sns.boxplot(data=city_data, x='City', y='Price', hue='Price Range', 
            palette='Set2', showfliers=False)
plt.title('Price Distribution by City (Top 8 Cities)', fontsize=12, fontweight='bold')
plt.ylabel('Price', fontsize=10)
plt.xlabel('City', fontsize=10)
plt.xticks(rotation=45, ha='right')
plt.yscale('log')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# 7. Covered Area vs Price scatter plot
plt.subplot(3, 3, 7)
if 'Covered Area' in df.columns:
    sample_data = df.sample(min(1000, len(df)))  # Sample for better visualization
    sns.scatterplot(data=sample_data, x='Covered Area', y='Price', hue='Price Range', 
                   palette='viridis', alpha=0.6)
    plt.title('Covered Area vs Price', fontsize=12, fontweight='bold')
    plt.xlabel('Covered Area (sq.ft.)', fontsize=10)
    plt.ylabel('Price', fontsize=10)
    plt.yscale('log')
else:
    plt.text(0.5, 0.5, 'Covered Area data not available', ha='center', va='center')
    plt.title('Covered Area Data Not Available', fontsize=12, fontweight='bold')

# 8. City-wise property count by price range
plt.subplot(3, 3, 8)
city_price_count = df.groupby(['City', 'Price Range']).size().unstack().fillna(0)
city_price_count['Total'] = city_price_count.sum(axis=1)
top_cities = city_price_count.nlargest(10, 'Total').drop('Total', axis=1)
top_cities.plot(kind='bar', stacked=True, colormap='tab10', ax=plt.gca())
plt.title('Top 10 Cities: Property Count by Price Range', fontsize=12, fontweight='bold')
plt.ylabel('Number of Properties', fontsize=10)
plt.xlabel('City', fontsize=10)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Price Range', bbox_to_anchor=(1.05, 1), loc='upper left')

# 9. Price range distribution pie chart
plt.subplot(3, 3, 9)
price_range_counts = df['Price Range'].value_counts()
colors = ['#ff9999', '#66b3ff', '#99ff99']
plt.pie(price_range_counts.values, labels=price_range_counts.index, autopct='%1.1f%%',
        colors=colors, startangle=90)
plt.title('Overall Price Range Distribution', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

# Generate comprehensive market summary
print("=" * 80)
print("COMPREHENSIVE REAL ESTATE MARKET ANALYSIS")
print("=" * 80)

for price_range in ['Low', 'Medium', 'High']:
    subset = df[df['Price Range'] == price_range]
    
    print(f"\n{'='*40}")
    print(f"{price_range.upper()} PRICE RANGE MARKET SUMMARY")
    print(f"{'='*40}")
    
    # Basic statistics
    print(f"📊 Basic Statistics:")
    print(f"   • Number of properties: {len(subset):,}")
    print(f"   • Average price: ₹{subset['Price'].mean():,.0f}")
    print(f"   • Price range: ₹{subset['Price'].min():,.0f} - ₹{subset['Price'].max():,.0f}")
    
    if 'Covered Area' in df.columns:
        print(f"   • Average covered area: {subset['Covered Area'].mean():.0f} sq.ft.")
    
    if 'bedroom' in df.columns:
        print(f"   • Average bedrooms: {subset['bedroom'].mean():.1f}")
    
    # Top cities
    print(f"\n🏙️  Geographic Distribution:")
    top_cities = subset['City'].value_counts().head(5)
    for city, count in top_cities.items():
        percentage = (count / len(subset)) * 100
        print(f"   • {city}: {count} properties ({percentage:.1f}%)")
    
    # Property type analysis
    print(f"\n🏠 Property Type Analysis:")
    prop_types = subset['Type of Property'].value_counts().head(3)
    for prop_type, count in prop_types.items():
        percentage = (count / len(subset)) * 100
        print(f"   • {prop_type}: {count} properties ({percentage:.1f}%)")
    
    # Amenities analysis
    print(f"\n⭐ Key Amenities (Availability > 30%):")
    amenities_to_check = ['Swimming Pool', 'Gymnasium', 'Park', 'Security', 'Lift', 'Club House']
    for amenity in amenities_to_check:
        if amenity in df.columns:
            availability = subset[amenity].mean() * 100
            if availability > 30:
                print(f"   • {amenity}: {availability:.1f}%")
    
    # Developer analysis if available
    if 'Developer' in df.columns:
        print(f"\n🏗️  Top Developers:")
        top_developers = subset['Developer'].value_counts().head(3)
        for developer, count in top_developers.items():
            print(f"   • {developer}: {count} properties")

# Additional market insights
print(f"\n{'='*80}")
print("OVERALL MARKET INSIGHTS")
print(f"{'='*80}")

# City analysis
city_stats = df.groupby('City').agg({
    'Price': ['count', 'mean', 'median'],
    'bedroom': 'mean' if 'bedroom' in df.columns else 'count'
}).round(0)

city_stats.columns = ['Property_Count', 'Avg_Price', 'Median_Price', 'Avg_Bedrooms']
city_stats = city_stats.sort_values('Avg_Price', ascending=False)

print(f"\n🏆 Premium Markets (Highest Average Prices):")
for city in city_stats.head(3).index:
    price = city_stats.loc[city, 'Avg_Price']
    print(f"   • {city}: ₹{price:,.0f}")

print(f"\n💰 Most Affordable Markets:")
for city in city_stats.tail(3).index:
    price = city_stats.loc[city, 'Avg_Price']
    print(f"   • {city}: ₹{price:,.0f}")

print(f"\n📈 Market Concentration:")
total_properties = len(df)
for price_range in ['Low', 'Medium', 'High']:
    count = len(df[df['Price Range'] == price_range])
    percentage = (count / total_properties) * 100
    print(f"   • {price_range} range: {count} properties ({percentage:.1f}%)")

# Save the analysis to CSV for further use
df.to_csv('real_estate_analysis_with_price_ranges.csv', index=False)
print(f"\n✅ Analysis complete! Data saved to 'real_estate_analysis_with_price_ranges.csv'")