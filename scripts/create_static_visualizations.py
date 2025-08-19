#!/usr/bin/env python3
"""
Port Drayage Static Visualization
Creates static charts and maps for quick analysis of port drayage data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

def load_data():
    """Load the port drayage CSV data."""
    try:
        df = pd.read_csv('data/port_drayage_dummy_data.csv')
        df['date'] = pd.to_datetime(df['date'])
        return df
    except FileNotFoundError:
        print("Error: Could not find data/port_drayage_dummy_data.csv")
        return None

def create_comprehensive_dashboard(df):
    """Create a comprehensive dashboard with multiple visualizations."""
    
    # Set up the style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    # 1. Geographic scatter plot (top left, spans 2x2) - IMPORTS ONLY
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    
    # Get import data only
    imports = df[df['order_type'] == 'import']
    
    # Find and highlight the main hub (port)
    origin_counts = df['origin_zip'].value_counts()
    dest_counts = df['destination_zip'].value_counts()
    all_zips = origin_counts.add(dest_counts, fill_value=0).sort_values(ascending=False)
    main_hub = all_zips.index[0]
    
    hub_data = df[df['origin_zip'] == main_hub]
    if len(hub_data) == 0:
        hub_data = df[df['destination_zip'] == main_hub]
        hub_lat, hub_lng = hub_data.iloc[0]['destination_lat'], hub_data.iloc[0]['destination_lng']
    else:
        hub_lat, hub_lng = hub_data.iloc[0]['origin_lat'], hub_data.iloc[0]['origin_lng']
    
    # Add concentric rings around the port
    from matplotlib.patches import Circle
    ring_distances = [50, 100, 150, 200]  # Miles represented as approximate coordinate distances
    ring_colors = ['lightblue', 'lightgreen', 'yellow', 'orange']
    coord_per_mile = 0.014  # Approximate conversion for Southern California
    
    for i, (distance, color) in enumerate(zip(ring_distances, ring_colors)):
        radius = distance * coord_per_mile
        circle = Circle((hub_lng, hub_lat), radius, fill=False, 
                       color=color, linewidth=2, alpha=0.7, linestyle='--')
        ax1.add_patch(circle)
        # Add distance labels
        ax1.text(hub_lng + radius * 0.7, hub_lat + radius * 0.7, 
                f'{distance}mi', fontsize=8, color=color, fontweight='bold')
    
    # Plot destinations colored by RPM (rate per mile)
    rpm_colors = plt.cm.viridis(plt.Normalize(vmin=df['RPM'].min(), vmax=df['RPM'].max())(imports['RPM']))
    scatter = ax1.scatter(imports['destination_lng'], imports['destination_lat'], 
                         c=imports['RPM'], cmap='viridis', s=30, alpha=0.8, 
                         edgecolors='white', linewidth=0.5)
    
    # Add colorbar for RPM
    cbar = plt.colorbar(scatter, ax=ax1, shrink=0.8)
    cbar.set_label('Rate Per Mile ($)', rotation=270, labelpad=20)
    
    # Highlight the port
    ax1.scatter(hub_lng, hub_lat, c='red', s=300, marker='*', 
               edgecolor='black', linewidth=3, label=f'Port Hub ({main_hub})', zorder=10)
    
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title('Import Destinations - Colored by Rate Per Mile', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Rate vs Distance (top right)
    ax2 = fig.add_subplot(gs[0, 2:])
    for order_type, color in [('import', 'blue'), ('export', 'red')]:
        data = df[df['order_type'] == order_type]
        ax2.scatter(data['miles'], data['rate'], c=color, alpha=0.6, s=20, label=order_type.title())
    
    # Add trendline
    z = np.polyfit(df['miles'], df['rate'], 1)
    p = np.poly1d(z)
    ax2.plot(df['miles'].sort_values(), p(df['miles'].sort_values()), "gray", linestyle='--', alpha=0.8)
    
    ax2.set_xlabel('Distance (miles)')
    ax2.set_ylabel('Rate ($)')
    ax2.set_title('Rate vs Distance by Order Type', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. RPM Distribution (middle right)
    ax3 = fig.add_subplot(gs[1, 2:])
    df.boxplot(column='RPM', by='order_type', ax=ax3)
    ax3.set_title('Rate Per Mile Distribution', fontweight='bold')
    ax3.set_xlabel('Order Type')
    ax3.set_ylabel('RPM ($)')
    plt.suptitle('')  # Remove automatic title
    
    # 4. Monthly trends (bottom left)
    ax4 = fig.add_subplot(gs[2, 0:2])
    df_monthly = df.copy()
    df_monthly['month'] = df_monthly['date'].dt.to_period('M')
    
    monthly_stats = df_monthly.groupby(['month', 'order_type']).agg({
        'rate': 'mean',
        'miles': 'mean',
        'RPM': 'mean'
    }).reset_index()
    
    for order_type, color in [('import', 'blue'), ('export', 'red')]:
        data = monthly_stats[monthly_stats['order_type'] == order_type]
        ax4.plot(data['month'].astype(str), data['rate'], marker='o', color=color, label=f'{order_type.title()} Rate', linewidth=2)
    
    ax4.set_xlabel('Month')
    ax4.set_ylabel('Average Rate ($)')
    ax4.set_title('Monthly Rate Trends', fontweight='bold')
    ax4.legend()
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # 5. Top carriers (bottom right)
    ax5 = fig.add_subplot(gs[2, 2:])
    top_carriers = df['carrier'].value_counts().head(10)
    bars = ax5.barh(range(len(top_carriers)), top_carriers.values, color='steelblue')
    ax5.set_yticks(range(len(top_carriers)))
    ax5.set_yticklabels([name[:25] + '...' if len(name) > 25 else name for name in top_carriers.index])
    ax5.set_xlabel('Number of Routes')
    ax5.set_title('Top 10 Carriers by Volume', fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax5.text(width + 1, bar.get_y() + bar.get_height()/2, 
                f'{int(width)}', ha='left', va='center', fontsize=9)
    
    # 6. Distance distribution (middle left)
    ax6 = fig.add_subplot(gs[3, 0:2])
    ax6.hist(df[df['order_type'] == 'import']['miles'], bins=30, alpha=0.7, color='blue', label='Import', density=True)
    ax6.hist(df[df['order_type'] == 'export']['miles'], bins=30, alpha=0.7, color='red', label='Export', density=True)
    ax6.set_xlabel('Distance (miles)')
    ax6.set_ylabel('Density')
    ax6.set_title('Distance Distribution by Order Type', fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Rate efficiency (bottom right)
    ax7 = fig.add_subplot(gs[3, 2:])
    
    # Create efficiency metric (higher RPM = more efficient)
    df_efficiency = df.copy()
    df_efficiency['efficiency_score'] = df_efficiency['RPM'] / df_efficiency['RPM'].max() * 100
    
    efficiency_by_distance = df_efficiency.groupby(pd.cut(df_efficiency['miles'], bins=5))['efficiency_score'].mean()
    distance_labels = [f"{int(interval.left)}-{int(interval.right)}" for interval in efficiency_by_distance.index]
    
    bars = ax7.bar(range(len(efficiency_by_distance)), efficiency_by_distance.values, color='green', alpha=0.7)
    ax7.set_xticks(range(len(efficiency_by_distance)))
    ax7.set_xticklabels(distance_labels, rotation=45)
    ax7.set_xlabel('Distance Range (miles)')
    ax7.set_ylabel('Efficiency Score (%)')
    ax7.set_title('Route Efficiency by Distance', fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2, height + 1, 
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Add overall title
    fig.suptitle('Port Drayage Analytics Dashboard', fontsize=20, fontweight='bold', y=0.98)
    
    # Save the dashboard
    plt.savefig('port_drayage_dashboard.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def create_summary_infographic(df):
    """Create a summary infographic with key statistics."""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Background
    bg = FancyBboxPatch((0.5, 0.5), 9, 9, boxstyle="round,pad=0.1", 
                       facecolor='lightblue', alpha=0.1, edgecolor='navy', linewidth=2)
    ax.add_patch(bg)
    
    # Title
    ax.text(5, 9.2, 'PORT DRAYAGE DATA SUMMARY', ha='center', va='center', 
           fontsize=24, fontweight='bold', color='navy')
    
    # Key statistics
    stats = [
        f"Total Routes: {len(df):,}",
        f"Date Range: {df['date'].min().strftime('%b %Y')} - {df['date'].max().strftime('%b %Y')}",
        f"Unique Carriers: {df['carrier'].nunique()}",
        f"Import Routes: {len(df[df['order_type'] == 'import']):,} ({len(df[df['order_type'] == 'import'])/len(df)*100:.1f}%)",
        f"Export Routes: {len(df[df['order_type'] == 'export']):,} ({len(df[df['order_type'] == 'export'])/len(df)*100:.1f}%)",
        f"Average Rate: ${df['rate'].mean():.2f}",
        f"Average Distance: {df['miles'].mean():.1f} miles",
        f"Average RPM: ${df['RPM'].mean():.2f}/mile"
    ]
    
    for i, stat in enumerate(stats):
        y_pos = 8.2 - i * 0.6
        ax.text(1, y_pos, stat, ha='left', va='center', fontsize=14, fontweight='bold')
    
    # Top carriers box
    top_carriers = df['carrier'].value_counts().head(5)
    ax.text(6, 8.2, 'TOP CARRIERS', ha='left', va='center', fontsize=16, fontweight='bold', color='navy')
    
    for i, (carrier, count) in enumerate(top_carriers.items()):
        y_pos = 7.6 - i * 0.4
        carrier_short = carrier[:30] + '...' if len(carrier) > 30 else carrier
        ax.text(6, y_pos, f"{i+1}. {carrier_short}: {count}", ha='left', va='center', fontsize=11)
    
    # Rate statistics box
    rate_stats = [
        f"Min Rate: ${df['rate'].min():.2f}",
        f"Max Rate: ${df['rate'].max():.2f}",
        f"Median Rate: ${df['rate'].median():.2f}",
        f"Std Dev: ${df['rate'].std():.2f}"
    ]
    
    ax.text(6, 5.2, 'RATE STATISTICS', ha='left', va='center', fontsize=16, fontweight='bold', color='navy')
    
    for i, stat in enumerate(rate_stats):
        y_pos = 4.6 - i * 0.4
        ax.text(6, y_pos, stat, ha='left', va='center', fontsize=11)
    
    plt.savefig('port_drayage_summary.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def main():
    """Main function to create static visualizations."""
    print("Creating Port Drayage Static Visualizations...")
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    print("Creating comprehensive dashboard...")
    create_comprehensive_dashboard(df)
    print("Dashboard saved as: port_drayage_dashboard.png")
    
    print("Creating summary infographic...")
    create_summary_infographic(df)
    print("Summary saved as: port_drayage_summary.png")
    
    print("\nStatic visualizations complete!")
    print("Files created:")
    print("- port_drayage_dashboard.png (comprehensive analytics)")
    print("- port_drayage_summary.png (key statistics)")

if __name__ == "__main__":
    main()
