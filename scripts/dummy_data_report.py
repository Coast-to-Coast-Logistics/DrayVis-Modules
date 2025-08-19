import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Load data
csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'port_drayage_dummy_data.csv')
df = pd.read_csv(csv_path)
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.to_period('M')

# Create distance buckets for analysis
df['distance_bucket'] = pd.cut(df['miles'], 
                              bins=[0, 25, 50, 75, 100, 125, 150, 175, 200], 
                              labels=['0-25', '25-50', '50-75', '75-100', '100-125', '125-150', '150-175', '175-200'])

# Charts folder setup
charts_dir = os.path.join(os.path.dirname(__file__), '..', 'charts')
os.makedirs(charts_dir, exist_ok=True)

print(f"📊 Analyzing {len(df)} drayage records...")
print(f"📏 Distance range: {df['miles'].min():.1f} - {df['miles'].max():.1f} miles")
print(f"💰 Rate range: ${df['rate'].min():.2f} - ${df['rate'].max():.2f}")
print(f"🚛 Average distance: {df['miles'].mean():.1f} miles")
print(f"💲 Average rate: ${df['rate'].mean():.2f}")

# Set style for better-looking charts
plt.style.use('seaborn-v0_8')
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83', '#048A81', '#54C6EB', '#F4A261']

# 1. Distance Distribution (Most Important Chart)
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
distance_counts = df['distance_bucket'].value_counts().sort_index()
bars = plt.bar(range(len(distance_counts)), distance_counts.values, color=colors[:len(distance_counts)])
plt.title('Distance Distribution by Bucket\n(Weighted Selection Results)', fontsize=14, fontweight='bold')
plt.xlabel('Distance Bucket (Miles)')
plt.ylabel('Number of Orders')
plt.xticks(range(len(distance_counts)), distance_counts.index, rotation=45)
# Add value labels on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 10, f'{height}', 
             ha='center', va='bottom', fontweight='bold')

# 2. Enhanced RPM vs Distance Analysis
plt.subplot(2, 2, 2)
scatter = plt.scatter(df['miles'], df['RPM'], alpha=0.6, c=df['miles'], cmap='viridis', s=20)
plt.colorbar(scatter, label='Distance (miles)')
plt.title('RPM vs Distance\n(Rate Per Mile Economics)', fontsize=14, fontweight='bold')
plt.xlabel('Distance (Miles)')
plt.ylabel('Rate Per Mile ($)')
# Add trend line
z = np.polyfit(df['miles'], df['RPM'], 1)
p = np.poly1d(z)
plt.plot(df['miles'], p(df['miles']), "r--", alpha=0.8, linewidth=2)

# 3. Average Rate by Distance Bucket
plt.subplot(2, 2, 3)
avg_rates = df.groupby('distance_bucket')['rate'].mean()
bars = plt.bar(range(len(avg_rates)), avg_rates.values, color=colors[:len(avg_rates)])
plt.title('Average Rate by Distance\n(Distance-Based Pricing)', fontsize=14, fontweight='bold')
plt.xlabel('Distance Bucket (Miles)')
plt.ylabel('Average Rate ($)')
plt.xticks(range(len(avg_rates)), avg_rates.index, rotation=45)
# Add value labels
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 5, f'${height:.0f}', 
             ha='center', va='bottom', fontweight='bold')

# 4. RPM Distribution with Better Context
plt.subplot(2, 2, 4)
plt.hist(df['RPM'], bins=30, color=colors[0], alpha=0.7, edgecolor='black')
plt.axvline(df['RPM'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ${df["RPM"].mean():.2f}')
plt.axvline(df['RPM'].median(), color='orange', linestyle='--', linewidth=2, label=f'Median: ${df["RPM"].median():.2f}')
plt.title('Rate Per Mile Distribution\n(Economic Analysis)', fontsize=14, fontweight='bold')
plt.xlabel('Rate Per Mile ($)')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(charts_dir, 'comprehensive_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()

# 5. Geographic Heat Map of Order Concentration
plt.figure(figsize=(15, 10))

# Distance bucket analysis with carrier insights
plt.subplot(2, 3, 1)
distance_rpm = df.groupby('distance_bucket')['RPM'].mean()
bars = plt.bar(range(len(distance_rpm)), distance_rpm.values, color=colors[:len(distance_rpm)])
plt.title('Average RPM by Distance\n(Rate Economics)', fontsize=12, fontweight='bold')
plt.xlabel('Distance Bucket')
plt.ylabel('Average RPM ($)')
plt.xticks(range(len(distance_rpm)), distance_rpm.index, rotation=45)
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.05, f'${height:.2f}', 
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# Rate progression over time
plt.subplot(2, 3, 2)
monthly = df.groupby('month')['rate'].mean()
monthly.plot(marker='o', linewidth=2, markersize=6, color=colors[1])
plt.title('Average Rate Over Time\n(Market Trends)', fontsize=12, fontweight='bold')
plt.ylabel('Average Rate ($)')
plt.xlabel('Month')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Top carriers by volume
plt.subplot(2, 3, 3)
top_carriers = df['carrier'].value_counts().head(10)
bars = plt.barh(range(len(top_carriers)), top_carriers.values, color=colors[:len(top_carriers)])
plt.title('Top 10 Carriers by Volume\n(Market Share)', fontsize=12, fontweight='bold')
plt.xlabel('Order Count')
plt.yticks(range(len(top_carriers)), [name[:20] + '...' if len(name) > 20 else name for name in top_carriers.index])
for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width + 5, bar.get_y() + bar.get_height()/2., f'{width}', 
             ha='left', va='center', fontweight='bold')

# Import vs Export distribution
plt.subplot(2, 3, 4)
order_counts = df['order_type'].value_counts()
colors_pie = [colors[0], colors[2]]
wedges, texts, autotexts = plt.pie(order_counts.values, labels=order_counts.index, 
                                  autopct='%1.1f%%', startangle=90, colors=colors_pie,
                                  textprops={'fontsize': 12, 'fontweight': 'bold'})
plt.title('Import vs Export Orders\n(Trade Balance)', fontsize=12, fontweight='bold')

# Distance vs Rate correlation
plt.subplot(2, 3, 5)
# Create hexbin plot for better visualization of dense data
plt.hexbin(df['miles'], df['rate'], gridsize=20, cmap='Blues', alpha=0.8)
plt.colorbar(label='Order Density')
plt.title('Distance vs Rate Correlation\n(Pricing Model)', fontsize=12, fontweight='bold')
plt.xlabel('Distance (Miles)')
plt.ylabel('Rate ($)')

# Economic analysis: Rate efficiency
plt.subplot(2, 3, 6)
df['rate_efficiency'] = df['rate'] / df['miles']  # Rate per mile efficiency
efficiency_buckets = df.groupby('distance_bucket')['rate_efficiency'].mean()
bars = plt.bar(range(len(efficiency_buckets)), efficiency_buckets.values, 
               color=colors[:len(efficiency_buckets)])
plt.title('Rate Efficiency by Distance\n($/Mile Analysis)', fontsize=12, fontweight='bold')
plt.xlabel('Distance Bucket')
plt.ylabel('Rate per Mile ($)')
plt.xticks(range(len(efficiency_buckets)), efficiency_buckets.index, rotation=45)
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'${height:.2f}', 
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(charts_dir, 'detailed_analytics.png'), dpi=300, bbox_inches='tight')
plt.close()

# Generate summary statistics
print("\n" + "="*60)
print("📈 DRAYAGE DATA ANALYTICS SUMMARY")
print("="*60)

print(f"\n🎯 DISTANCE ANALYSIS:")
print(f"   • Total Orders: {len(df):,}")
print(f"   • Distance Range: {df['miles'].min():.1f} - {df['miles'].max():.1f} miles")
print(f"   • Average Distance: {df['miles'].mean():.1f} miles")
print(f"   • Median Distance: {df['miles'].median():.1f} miles")

distance_dist = df['distance_bucket'].value_counts().sort_index()
print(f"\n📊 DISTANCE BUCKET DISTRIBUTION:")
for bucket, count in distance_dist.items():
    percentage = (count / len(df)) * 100
    print(f"   • {bucket} miles: {count:,} orders ({percentage:.1f}%)")

print(f"\n💰 RATE ANALYSIS:")
print(f"   • Rate Range: ${df['rate'].min():.2f} - ${df['rate'].max():.2f}")
print(f"   • Average Rate: ${df['rate'].mean():.2f}")
print(f"   • Median Rate: ${df['rate'].median():.2f}")

print(f"\n🚛 RPM (RATE PER MILE) ANALYSIS:")
print(f"   • RPM Range: ${df['RPM'].min():.2f} - ${df['RPM'].max():.2f}")
print(f"   • Average RPM: ${df['RPM'].mean():.2f}")
print(f"   • Median RPM: ${df['RPM'].median():.2f}")

rpm_by_distance = df.groupby('distance_bucket')['RPM'].mean()
print(f"\n📏 AVERAGE RPM BY DISTANCE:")
for bucket, rpm in rpm_by_distance.items():
    print(f"   • {bucket} miles: ${rpm:.2f}/mile")

print(f"\n🏢 CARRIER ANALYSIS:")
top_5_carriers = df['carrier'].value_counts().head(5)
for i, (carrier, count) in enumerate(top_5_carriers.items(), 1):
    percentage = (count / len(df)) * 100
    print(f"   {i}. {carrier}: {count} orders ({percentage:.1f}%)")

print(f"\n📦 ORDER TYPE DISTRIBUTION:")
order_dist = df['order_type'].value_counts()
for order_type, count in order_dist.items():
    percentage = (count / len(df)) * 100
    print(f"   • {order_type.title()}: {count:,} orders ({percentage:.1f}%)")

print(f"\n✅ WEIGHTED BUCKET SUCCESS:")
long_distance = df[df['miles'] > 100].shape[0]
long_distance_pct = (long_distance / len(df)) * 100
print(f"   • Orders > 100 miles: {long_distance:,} ({long_distance_pct:.1f}%)")
very_long_distance = df[df['miles'] > 150].shape[0]
very_long_distance_pct = (very_long_distance / len(df)) * 100
print(f"   • Orders > 150 miles: {very_long_distance:,} ({very_long_distance_pct:.1f}%)")
print(f"   • Maximum distance achieved: {df['miles'].max():.1f} miles")

print("\n" + "="*60)
print("📁 CHARTS GENERATED:")
print("="*60)
print(f"📊 {os.path.join(charts_dir, 'comprehensive_analysis.png')}")
print(f"📈 {os.path.join(charts_dir, 'detailed_analytics.png')}")
print("\n🎉 Analysis complete! Check the charts folder for visualizations.")
print("="*60)
