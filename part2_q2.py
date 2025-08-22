import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import warnings
warnings.filterwarnings('ignore')

# Set the style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load the dataset
df = pd.read_csv('M25_DA_A1_Dataset2.csv')

# Data cleaning and preparation
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
df = df.dropna(subset=['Price', 'City'])

# Filter for Mumbai and Thane
cities_to_analyze = ['Mumbai', 'Thane']
df_cities = df[df['City'].isin(cities_to_analyze)].copy()

# Convert carpet area to numeric if available
if 'Carpet Area' in df_cities.columns:
    df_cities['Carpet Area'] = pd.to_numeric(df_cities['Carpet Area'], errors='coerce')

# Clean Commercial column - handle both numeric and string values
if 'Commercial' in df_cities.columns:
    # Convert to numeric, handling errors
    df_cities['Commercial'] = pd.to_numeric(df_cities['Commercial'], errors='coerce')
    # Fill NaN values with 0 (assuming residential if not specified)
    df_cities['Commercial'] = df_cities['Commercial'].fillna(0)
    # Convert to binary (0 = Residential, 1 = Commercial)
    df_cities['Commercial'] = df_cities['Commercial'].apply(lambda x: 1 if x > 0 else 0)
else:
    # Create Commercial column based on property type if not available
    df_cities['Commercial'] = df_cities['Type of Property'].apply(
        lambda x: 1 if isinstance(x, str) and any(word in x.lower() for word in ['commercial', 'office', 'shop', 'retail', 'business']) else 0
    )

# Create subplots
fig = plt.figure(figsize=(25, 20))
# plt.suptitle('MUMBAI vs THANE: COMPREHENSIVE REAL ESTATE INVESTMENT ANALYSIS', 
#              fontsize=20, fontweight='bold', y=0.98)

# 1. Price Distribution Comparison
plt.subplot(3, 4, 1)
sns.boxplot(data=df_cities, x='City', y='Price', showfliers=False, palette='Set2')
plt.title('Price Distribution Comparison\n(Mumbai vs Thane)', fontsize=14, fontweight='bold')
plt.ylabel('Price (₹)', fontsize=12)
plt.xlabel('City', fontsize=12)
plt.yscale('log')
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'₹{x/1000000:.0f}M'))
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.grid(True, alpha=0.3)

# 2. Property Type Distribution
plt.subplot(3, 4, 2)
if 'Type of Property' in df_cities.columns:
    prop_type_dist = pd.crosstab(df_cities['City'], df_cities['Type of Property'], normalize='index') * 100
    # Get top 5 property types for better visualization
    top_properties = prop_type_dist.sum().nlargest(5).index
    prop_type_dist = prop_type_dist[top_properties]
    prop_type_dist.plot(kind='bar', stacked=True, ax=plt.gca(), colormap='tab10')
    plt.title('Property Type Distribution (%)', fontsize=14, fontweight='bold')
    plt.ylabel('Percentage', fontsize=12)
    plt.xlabel('City', fontsize=12)
    plt.legend(title='Property Type', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
else:
    plt.text(0.5, 0.5, 'Property Type data not available', ha='center', va='center', fontsize=12)
    plt.title('Property Type Data Not Available', fontsize=14, fontweight='bold')

# 3. Average Price by Property Type
plt.subplot(3, 4, 3)
if 'Type of Property' in df_cities.columns:
    avg_price_type = df_cities.groupby(['City', 'Type of Property'])['Price'].mean().unstack()
    # Get top 5 property types by count for better visualization
    top_properties = df_cities['Type of Property'].value_counts().head(5).index
    avg_price_type = avg_price_type[top_properties]
    avg_price_type.plot(kind='bar', ax=plt.gca(), colormap='viridis')
    plt.title('Average Price by Property Type', fontsize=14, fontweight='bold')
    plt.ylabel('Average Price (₹)', fontsize=12)
    plt.xlabel('City', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yscale('log')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'₹{x/1000000:.1f}M'))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3, axis='y')
else:
    plt.text(0.5, 0.5, 'Property Type data not available', ha='center', va='center', fontsize=12)
    plt.title('Property Type Data Not Available', fontsize=14, fontweight='bold')

# 4. Carpet Area Distribution
plt.subplot(3, 4, 4)
if 'Carpet Area' in df_cities.columns:
    carpet_data = df_cities.dropna(subset=['Carpet Area'])
    if len(carpet_data) > 0:
        sns.boxplot(data=carpet_data, x='City', y='Carpet Area', showfliers=False, palette='Set3')
        plt.title('Carpet Area Distribution', fontsize=14, fontweight='bold')
        plt.ylabel('Carpet Area (sq.ft)', fontsize=12)
        plt.xlabel('City', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
    else:
        plt.text(0.5, 0.5, 'No valid Carpet Area data', ha='center', va='center', fontsize=12)
        plt.title('Carpet Area Data Not Available', fontsize=14, fontweight='bold')
else:
    plt.text(0.5, 0.5, 'Carpet Area data not available', ha='center', va='center', fontsize=12)
    plt.title('Carpet Area Data Not Available', fontsize=14, fontweight='bold')

# 5. Price per Sq.ft Comparison
plt.subplot(3, 4, 5)
if 'Carpet Area' in df_cities.columns:
    carpet_data = df_cities.dropna(subset=['Carpet Area'])
    if len(carpet_data) > 0:
        carpet_data['Price_per_sqft'] = carpet_data['Price'] / carpet_data['Carpet Area']
        # Remove extreme outliers for better visualization
        carpet_data = carpet_data[carpet_data['Price_per_sqft'] < carpet_data['Price_per_sqft'].quantile(0.99)]
        sns.boxplot(data=carpet_data, x='City', y='Price_per_sqft', showfliers=False, palette='pastel')
        plt.title('Price per Sq.ft Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('Price per Sq.ft (₹)', fontsize=12)
        plt.xlabel('City', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
    else:
        plt.text(0.5, 0.5, 'No valid data for calculation', ha='center', va='center', fontsize=12)
        plt.title('Price per Sq.ft Not Available', fontsize=14, fontweight='bold')
else:
    plt.text(0.5, 0.5, 'Carpet Area data not available', ha='center', va='center', fontsize=12)
    plt.title('Price per Sq.ft Not Available', fontsize=14, fontweight='bold')

# 6. Bedroom Distribution
plt.subplot(3, 4, 6)
if 'bedroom' in df_cities.columns:
    bedroom_dist = pd.crosstab(df_cities['City'], df_cities['bedroom'], normalize='index') * 100
    bedroom_dist.plot(kind='bar', stacked=True, ax=plt.gca(), colormap='cool')
    plt.title('Bedroom Distribution (%)', fontsize=14, fontweight='bold')
    plt.ylabel('Percentage', fontsize=12)
    plt.xlabel('City', fontsize=12)
    plt.legend(title='Bedrooms', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
else:
    plt.text(0.5, 0.5, 'Bedroom data not available', ha='center', va='center', fontsize=12)
    plt.title('Bedroom Data Not Available', fontsize=14, fontweight='bold')

# 7. Commercial vs Residential Analysis - FIXED VERSION
plt.subplot(3, 4, 7)
commercial_dist = pd.crosstab(df_cities['City'], df_cities['Commercial'], normalize='index') * 100

# Handle case where there might be only one category
if len(commercial_dist.columns) == 1:
    # If only one category exists, create the missing category with zeros
    missing_category = 1 if 0 in commercial_dist.columns else 0
    commercial_dist[missing_category] = 0
    commercial_dist = commercial_dist[[0, 1]]  # Ensure correct order

commercial_dist.columns = ['Residential', 'Commercial']
commercial_dist.plot(kind='bar', ax=plt.gca(), color=['skyblue', 'orange'])
plt.title('Commercial vs Residential Distribution (%)', fontsize=14, fontweight='bold')
plt.ylabel('Percentage', fontsize=12)
plt.xlabel('City', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3, axis='y')

# 8. Price Range Distribution
plt.subplot(3, 4, 8)
price_ranges = pd.qcut(df_cities['Price'], q=4, labels=['Budget', 'Mid-Range', 'Premium', 'Luxury'])
price_range_dist = pd.crosstab(df_cities['City'], price_ranges, normalize='index') * 100
price_range_dist.plot(kind='bar', ax=plt.gca(), colormap='plasma')
plt.title('Price Range Distribution (%)', fontsize=14, fontweight='bold')
plt.ylabel('Percentage', fontsize=12)
plt.xlabel('City', fontsize=12)
plt.legend(title='Price Range', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3, axis='y')

# 9. Scatter Plot: Price vs Carpet Area
plt.subplot(3, 4, 9)
if 'Carpet Area' in df_cities.columns:
    scatter_data = df_cities.dropna(subset=['Carpet Area']).sample(min(1000, len(df_cities)), random_state=42)
    sns.scatterplot(data=scatter_data, x='Carpet Area', y='Price', hue='City', alpha=0.7, palette='Set1', s=60)
    plt.title('Price vs Carpet Area', fontsize=14, fontweight='bold')
    plt.xlabel('Carpet Area (sq.ft)', fontsize=12)
    plt.ylabel('Price (₹)', fontsize=12)
    plt.yscale('log')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'₹{x/1000000:.0f}M'))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)
else:
    plt.text(0.5, 0.5, 'Carpet Area data not available', ha='center', va='center', fontsize=12)
    plt.title('Carpet Area Data Not Available', fontsize=14, fontweight='bold')

# 10. Top Locations within each city
plt.subplot(3, 4, 10)
if 'Area Name' in df_cities.columns:
    top_locations = df_cities.groupby(['City', 'Area Name']).size()
    top_locations = top_locations.groupby('City').nlargest(3).reset_index(level=0, drop=True).unstack()
    top_locations.plot(kind='bar', ax=plt.gca(), colormap='Set3')
    plt.title('Top 3 Locations by Property Count', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Properties', fontsize=12)
    plt.xlabel('City', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3, axis='y')
else:
    plt.text(0.5, 0.5, 'Area Name data not available', ha='center', va='center', fontsize=12)
    plt.title('Location Data Not Available', fontsize=14, fontweight='bold')

# 11. Amenities Comparison
plt.subplot(3, 4, 11)
amenities = ['Swimming Pool', 'Gymnasium', 'Park', 'Security', 'Lift', 'Club House']
available_amenities = [a for a in amenities if a in df_cities.columns]

if available_amenities:
    amenity_rates = df_cities.groupby('City')[available_amenities].mean() * 100
    amenity_rates.T.plot(kind='bar', ax=plt.gca())
    plt.title('Amenity Availability Comparison (%)', fontsize=14, fontweight='bold')
    plt.ylabel('Availability (%)', fontsize=12)
    plt.xlabel('Amenities', fontsize=12)
    plt.legend(title='City', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
else:
    plt.text(0.5, 0.5, 'Amenity data not available', ha='center', va='center', fontsize=12)
    plt.title('Amenity Data Not Available', fontsize=14, fontweight='bold')

# 12. Investment Value Heatmap
plt.subplot(3, 4, 12)
if 'Carpet Area' in df_cities.columns and 'Type of Property' in df_cities.columns:
    investment_data = df_cities.dropna(subset=['Carpet Area'])
    if len(investment_data) > 0:
        investment_data['Price_Area_Ratio'] = investment_data['Price'] / investment_data['Carpet Area']
        # Remove outliers for better heatmap
        investment_data = investment_data[investment_data['Price_Area_Ratio'] < investment_data['Price_Area_Ratio'].quantile(0.95)]
        investment_heatmap = investment_data.groupby(['City', 'Type of Property'])['Price_Area_Ratio'].median().unstack()
        # Get top property types
        top_properties = investment_heatmap.count().nlargest(5).index
        investment_heatmap = investment_heatmap[top_properties]
        sns.heatmap(investment_heatmap, annot=True, fmt='.0f', cmap='YlOrRd', 
                    cbar_kws={'label': 'Price per Sq.ft (₹)'})
        plt.title('Investment Value Heatmap\n(Median Price per Sq.ft)', fontsize=14, fontweight='bold')
        plt.xlabel('Property Type', fontsize=12)
        plt.ylabel('City', fontsize=12)
    else:
        plt.text(0.5, 0.5, 'No valid data for heatmap', ha='center', va='center', fontsize=12)
        plt.title('Investment Analysis Not Available', fontsize=14, fontweight='bold')
else:
    plt.text(0.5, 0.5, 'Required data not available', ha='center', va='center', fontsize=12)
    plt.title('Investment Analysis Not Available', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('mumbai_thane_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Generate detailed investment analysis report
print("=" * 80)
print("MUMBAI vs THANE: COMPREHENSIVE INVESTMENT ANALYSIS")
print("=" * 80)

for city in ['Mumbai', 'Thane']:
    city_data = df_cities[df_cities['City'] == city]
    
    print(f"\n{'='*40}")
    print(f"{city.upper()} - INVESTMENT PROFILE")
    print(f"{'='*40}")
    
    print(f"📊 Basic Statistics:")
    print(f"   • Total properties: {len(city_data):,}")
    print(f"   • Average price: ₹{city_data['Price'].mean():,.0f}")
    print(f"   • Median price: ₹{city_data['Price'].median():,.0f}")
    
    if 'Carpet Area' in city_data.columns:
        carpet_data = city_data.dropna(subset=['Carpet Area'])
        if len(carpet_data) > 0:
            print(f"   • Average carpet area: {carpet_data['Carpet Area'].mean():.0f} sq.ft")
            price_per_sqft = carpet_data['Price'].mean() / carpet_data['Carpet Area'].mean()
            print(f"   • Average price per sq.ft: ₹{price_per_sqft:.0f}")
    
    # Property type analysis
    print(f"\n🏠 Property Type Distribution:")
    if 'Type of Property' in city_data.columns:
        prop_types = city_data['Type of Property'].value_counts().head(5)
        for prop_type, count in prop_types.items():
            percentage = (count / len(city_data)) * 100
            print(f"   • {prop_type}: {count} properties ({percentage:.1f}%)")
    
    # Commercial vs Residential
    print(f"\n🏢 Commercial vs Residential:")
    commercial_count = city_data['Commercial'].sum()
    residential_count = len(city_data) - commercial_count
    print(f"   • Residential: {residential_count} ({residential_count/len(city_data)*100:.1f}%)")
    print(f"   • Commercial: {commercial_count} ({commercial_count/len(city_data)*100:.1f}%)")

print(f"\n✅ Analysis complete! Visualizations saved as 'mumbai_thane_comparison.png'")