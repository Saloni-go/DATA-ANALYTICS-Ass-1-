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
df = df.dropna(subset=['Price', 'City', 'Area Name'])

# Convert carpet area to numeric if available
if 'Carpet Area' in df.columns:
    df['Carpet Area'] = pd.to_numeric(df['Carpet Area'], errors='coerce')

# Define prime locations based on the actual area names in your data
prime_locations = {
    'Mumbai': [
        'South Mumbai', 'Bandra', 'Juhu', 'Powai', 'Lower Parel', 'Worli', 'Malabar Hill',
        'Marine Lines', 'Churchgate', 'Nariman Point', 'Colaba', 'Cuffe Parade', 'Pedder Road',
        'Altamount Road', 'Breach Candy', 'Carmichael Road', 'Nepean Sea Road', 'Lokhandwala Complex',
        'Versova', 'Andheri West', 'Khar West', 'Santacruz West', 'Vile Parle West', 'Goregaon West',
        'Juhu Scheme', 'Pali Hill', 'Turner Road', 'Carter Road', 'Linking Road', 'Hill Road'
    ],
    'Thane': [
        'Hiranandani Estate', 'Kolshet Road', 'Manpada', 'Vrindavan', 'Pokhran Road', 'Ghodbunder Road',
        'Majiwada', 'Khopat', 'Naupada', 'Vartak Nagar', 'Panch Pakhdi', 'Waghbil', 'Brahmand',
        'Lok Puram', 'Louis Wadi', 'Patlipada', 'Anand Nagar', 'Vasant Vihar', 'Bhayandarpada',
        'Kasarvadavali', 'Ovala', 'Kavesar', 'Shil Phata', 'Mumbra', 'Diva', 'Kalwa', 'Airoli'
    ]
}

# Enhanced prime location identification function
def is_prime_location(row):
    city = row['City']
    area_name = str(row['Area Name']).strip()
    
    if city not in prime_locations:
        return False
    
    # Check for exact matches
    if area_name in prime_locations[city]:
        return True
    
    # Check for partial matches
    area_lower = area_name.lower()
    for prime_area in prime_locations[city]:
        if prime_area.lower() in area_lower or area_lower in prime_area.lower():
            return True
    
    # Special cases for known premium areas
    premium_indicators = [
        'hiranandani', 'estate', 'pokhran', 'ghodbunder', 'kolshet', 
        'manpada', 'vrindavan', 'majiwada', 'naupada', 'vartak'
    ]
    
    if any(indicator in area_lower for indicator in premium_indicators):
        return True
    
    return False

df['Is_Prime_Location'] = df.apply(is_prime_location, axis=1)

# Convert all amenity columns to numeric (0/1)
amenity_columns = [
    'Swimming Pool', 'Gymnasium', 'Park', 'Security', 'Lift', 'Club House',
    'Power Back Up', 'Rain Water Harvesting', 'Parking', 'Concierge Services',
    'Marble flooring', 'Modular Kitchen', 'Private pool', 'Rera', 'Water Storage'
]

for amenity in amenity_columns:
    if amenity in df.columns:
        df[amenity] = pd.to_numeric(df[amenity], errors='coerce')
        df[amenity] = df[amenity].fillna(0).apply(lambda x: 1 if x > 0 else 0)

# Calculate price per sqft if carpet area is available
if 'Carpet Area' in df.columns:
    df['Price_per_sqft'] = df['Price'] / df['Carpet Area']
    # Remove extreme outliers
    df = df[df['Price_per_sqft'] < df['Price_per_sqft'].quantile(0.99)]

# Define high-budget properties (top 25% by price in each city)
df['Is_High_Budget'] = df.groupby('City')['Price'].transform(
    lambda x: x > x.quantile(0.75)
)

# Create the target group: High-budget prime location properties
df['Target_Group'] = np.where(
    df['Is_High_Budget'] & df['Is_Prime_Location'], 
    'Prime High-Budget',
    np.where(
        df['Is_High_Budget'] & ~df['Is_Prime_Location'],
        'Non-Prime High-Budget',
        'Other'
    )
)

# Filter to only include high-budget properties for analysis
analysis_df = df[df['Is_High_Budget']].copy()

# Get available amenities after conversion
available_amenities = [a for a in amenity_columns if a in analysis_df.columns]

# Create visualizations
fig = plt.figure(figsize=(25, 20))
# plt.suptitle('LOCATION-BASED PREMIUM ANALYSIS: PRIME vs NON-PRIME HIGH-BUDGET PROPERTIES', 
            #  fontsize=20, fontweight='bold', y=0.98)

# 1. Price Comparison by City and Location Type
plt.subplot(3, 4, 1)
price_comparison = analysis_df.groupby(['City', 'Target_Group'])['Price'].median().unstack()
cities_with_data = price_comparison.dropna().index
price_comparison = price_comparison.loc[cities_with_data]

if not price_comparison.empty:
    price_comparison.plot(kind='bar', ax=plt.gca(), color=['#FF6B6B', '#4ECDC4'])
    plt.title('Median Price: Prime vs Non-Prime Locations\n(High-Budget Properties)', fontsize=14, fontweight='bold')
    plt.ylabel('Median Price (₹)', fontsize=12)
    plt.xlabel('City', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yscale('log')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'₹{x/1000000:.1f}M'))
    plt.legend(title='Location Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3, axis='y')
else:
    plt.text(0.5, 0.5, 'Insufficient data for price comparison', ha='center', va='center', fontsize=12)
    plt.title('Price Comparison Not Available', fontsize=14, fontweight='bold')

# 2. Price per Sq.ft Comparison
plt.subplot(3, 4, 2)
if 'Price_per_sqft' in analysis_df.columns:
    sqft_comparison = analysis_df.groupby(['City', 'Target_Group'])['Price_per_sqft'].median().unstack()
    sqft_comparison = sqft_comparison.loc[cities_with_data]
    
    if not sqft_comparison.empty:
        sqft_comparison.plot(kind='bar', ax=plt.gca(), color=['#FF6B6B', '#4ECDC4'])
        plt.title('Price per Sq.ft: Prime vs Non-Prime\n(High-Budget Properties)', fontsize=14, fontweight='bold')
        plt.ylabel('Price per Sq.ft (₹)', fontsize=12)
        plt.xlabel('City', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Location Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3, axis='y')
    else:
        plt.text(0.5, 0.5, 'Insufficient data for sq.ft comparison', ha='center', va='center', fontsize=12)
        plt.title('Sq.ft Comparison Not Available', fontsize=14, fontweight='bold')
else:
    plt.text(0.5, 0.5, 'Carpet Area data not available', ha='center', va='center', fontsize=12)
    plt.title('Price per Sq.ft Not Available', fontsize=14, fontweight='bold')

# 3. Carpet Area Comparison
plt.subplot(3, 4, 3)
if 'Carpet Area' in analysis_df.columns:
    area_comparison = analysis_df.groupby(['City', 'Target_Group'])['Carpet Area'].median().unstack()
    area_comparison = area_comparison.loc[cities_with_data]
    
    if not area_comparison.empty:
        area_comparison.plot(kind='bar', ax=plt.gca(), color=['#FF6B6B', '#4ECDC4'])
        plt.title('Median Carpet Area: Prime vs Non-Prime\n(High-Budget Properties)', fontsize=14, fontweight='bold')
        plt.ylabel('Carpet Area (sq.ft)', fontsize=12)
        plt.xlabel('City', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Location Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3, axis='y')
    else:
        plt.text(0.5, 0.5, 'Insufficient data for area comparison', ha='center', va='center', fontsize=12)
        plt.title('Area Comparison Not Available', fontsize=14, fontweight='bold')
else:
    plt.text(0.5, 0.5, 'Carpet Area data not available', ha='center', va='center', fontsize=12)
    plt.title('Area Comparison Not Available', fontsize=14, fontweight='bold')

# 4. Premium Percentage by City
plt.subplot(3, 4, 4)
if 'Price_per_sqft' in analysis_df.columns and not sqft_comparison.empty:
    premium_data = []
    for city in cities_with_data:
        if city in sqft_comparison.index:
            prime_price = sqft_comparison.loc[city, 'Prime High-Budget']
            non_prime_price = sqft_comparison.loc[city, 'Non-Prime High-Budget']
            if not pd.isna(prime_price) and not pd.isna(non_prime_price) and non_prime_price > 0:
                premium_pct = ((prime_price - non_prime_price) / non_prime_price) * 100
                premium_data.append({'City': city, 'Premium_Percentage': premium_pct})
    
    if premium_data:
        premium_df = pd.DataFrame(premium_data)
        plt.bar(premium_df['City'], premium_df['Premium_Percentage'], color='#6A0DAD')
        plt.title('Location Premium Percentage\n(Prime vs Non-Prime Price per Sq.ft)', fontsize=14, fontweight='bold')
        plt.ylabel('Premium Percentage (%)', fontsize=12)
        plt.xlabel('City', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, v in enumerate(premium_df['Premium_Percentage']):
            plt.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
    else:
        plt.text(0.5, 0.5, 'Insufficient data for premium calculation', ha='center', va='center', fontsize=12)
        plt.title('Premium Analysis Not Available', fontsize=14, fontweight='bold')
else:
    plt.text(0.5, 0.5, 'Price data not available for premium calculation', ha='center', va='center', fontsize=12)
    plt.title('Premium Analysis Not Available', fontsize=14, fontweight='bold')

# 5. Amenities Comparison Heatmap
plt.subplot(3, 4, 5)
if available_amenities:
    # Ensure all amenity columns are numeric
    for amenity in available_amenities:
        analysis_df[amenity] = pd.to_numeric(analysis_df[amenity], errors='coerce').fillna(0)
    
    amenity_comparison = analysis_df.groupby('Target_Group')[available_amenities].mean() * 100
    # Filter only the groups we want to compare
    amenity_comparison = amenity_comparison.loc[['Prime High-Budget', 'Non-Prime High-Budget']]
    
    sns.heatmap(amenity_comparison, annot=True, fmt='.1f', cmap='YlOrRd', 
                cbar_kws={'label': 'Availability (%)'})
    plt.title('Amenity Availability Comparison\n(High-Budget Properties)', fontsize=14, fontweight='bold')
    plt.xlabel('Amenities', fontsize=12)
    plt.ylabel('Location Type', fontsize=12)
    plt.xticks(rotation=45, ha='right')
else:
    plt.text(0.5, 0.5, 'Amenity data not available', ha='center', va='center', fontsize=12)
    plt.title('Amenity Comparison Not Available', fontsize=14, fontweight='bold')

# 6. Property Type Distribution
plt.subplot(3, 4, 6)
if 'Type of Property' in analysis_df.columns:
    prop_type_dist = pd.crosstab(analysis_df['Target_Group'], analysis_df['Type of Property'], normalize='index') * 100
    # Filter only the groups we want to compare
    prop_type_dist = prop_type_dist.loc[['Prime High-Budget', 'Non-Prime High-Budget']]
    
    # Get top 5 property types
    top_properties = prop_type_dist.sum().nlargest(5).index
    prop_type_dist = prop_type_dist[top_properties]
    
    prop_type_dist.plot(kind='bar', stacked=True, ax=plt.gca(), colormap='tab10')
    plt.title('Property Type Distribution (%)', fontsize=14, fontweight='bold')
    plt.ylabel('Percentage', fontsize=12)
    plt.xlabel('Location Type', fontsize=12)
    plt.legend(title='Property Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
else:
    plt.text(0.5, 0.5, 'Property type data not available', ha='center', va='center', fontsize=12)
    plt.title('Property Type Comparison Not Available', fontsize=14, fontweight='bold')

# 7. Bedroom Distribution
plt.subplot(3, 4, 7)
if 'bedroom' in analysis_df.columns:
    bedroom_dist = pd.crosstab(analysis_df['Target_Group'], analysis_df['bedroom'], normalize='index') * 100
    # Filter only the groups we want to compare
    bedroom_dist = bedroom_dist.loc[['Prime High-Budget', 'Non-Prime High-Budget']]
    
    bedroom_dist.plot(kind='bar', stacked=True, ax=plt.gca(), colormap='cool')
    plt.title('Bedroom Distribution (%)', fontsize=14, fontweight='bold')
    plt.ylabel('Percentage', fontsize=12)
    plt.xlabel('Location Type', fontsize=12)
    plt.legend(title='Bedrooms', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
else:
    plt.text(0.5, 0.5, 'Bedroom data not available', ha='center', va='center', fontsize=12)
    plt.title('Bedroom Comparison Not Available', fontsize=14, fontweight='bold')

# 8. Luxury Amenities Comparison
plt.subplot(3, 4, 8)
luxury_amenities = ['Swimming Pool', 'Gymnasium', 'Club House', 'Concierge Services', 
                   'Marble flooring', 'Modular Kitchen', 'Private pool']
available_luxury = [a for a in luxury_amenities if a in analysis_df.columns]

if available_luxury:
    # Ensure all luxury amenity columns are numeric
    for amenity in available_luxury:
        analysis_df[amenity] = pd.to_numeric(analysis_df[amenity], errors='coerce').fillna(0)
    
    luxury_comparison = analysis_df.groupby('Target_Group')[available_luxury].mean() * 100
    # Filter only the groups we want to compare
    luxury_comparison = luxury_comparison.loc[['Prime High-Budget', 'Non-Prime High-Budget']]
    
    luxury_comparison.T.plot(kind='bar', ax=plt.gca(), color=['#FF6B6B', '#4ECDC4'])
    plt.title('Luxury Amenities Availability (%)', fontsize=14, fontweight='bold')
    plt.ylabel('Availability (%)', fontsize=12)
    plt.xlabel('Luxury Amenities', fontsize=12)
    plt.legend(title='Location Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
else:
    plt.text(0.5, 0.5, 'Luxury amenity data not available', ha='center', va='center', fontsize=12)
    plt.title('Luxury Amenities Not Available', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('location_premium_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Generate detailed analysis report
print("=" * 80)
print("LOCATION-BASED PREMIUM ANALYSIS REPORT")
print("=" * 80)

# Overall statistics
prime_count = len(analysis_df[analysis_df['Target_Group'] == 'Prime High-Budget'])
non_prime_count = len(analysis_df[analysis_df['Target_Group'] == 'Non-Prime High-Budget'])

print(f"📊 Overall Statistics:")
print(f"   • Prime High-Budget Properties: {prime_count:,}")
print(f"   • Non-Prime High-Budget Properties: {non_prime_count:,}")
print(f"   • Total High-Budget Properties Analyzed: {prime_count + non_prime_count:,}")

# City-wise analysis
print(f"\n🏙️  City-wise Analysis:")
for city in cities_with_data:
    city_data = analysis_df[analysis_df['City'] == city]
    prime_city = city_data[city_data['Target_Group'] == 'Prime High-Budget']
    non_prime_city = city_data[city_data['Target_Group'] == 'Non-Prime High-Budget']
    
    if len(prime_city) > 0 and len(non_prime_city) > 0:
        print(f"\n{city.upper()}:")
        print(f"   • Prime Properties: {len(prime_city):,}")
        print(f"   • Non-Prime Properties: {len(non_prime_city):,}")
        
        if 'Price_per_sqft' in analysis_df.columns:
            prime_price = prime_city['Price_per_sqft'].median()
            non_prime_price = non_prime_city['Price_per_sqft'].median()
            premium_pct = ((prime_price - non_prime_price) / non_prime_price) * 100
            
            print(f"   • Price Premium: {premium_pct:.1f}%")
            print(f"   • Prime Price/sq.ft: ₹{prime_price:.0f}")
            print(f"   • Non-Prime Price/sq.ft: ₹{non_prime_price:.0f}")

print(f"\n✅ Analysis complete! Visualizations saved as 'location_premium_analysis.png'")