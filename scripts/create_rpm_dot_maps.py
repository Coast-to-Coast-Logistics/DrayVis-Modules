#!/usr/bin/env python3
"""
Simple Port Drayage Dot Map with RPM Coloring and Concentric Rings
Creates exactly what was requested: dots colored by RPM, concentric rings, no lines, import/export capability.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
import seaborn as sns

def load_data():
    """Load the port drayage CSV data."""
    try:
        df = pd.read_csv('data/port_drayage_dummy_data.csv')
        df['date'] = pd.to_datetime(df['date'])
        return df
    except FileNotFoundError:
        print("Error: Could not find data/port_drayage_dummy_data.csv")
        return None

def create_dot_map_with_rings(df, order_type='import'):
    """Create a dot map with RPM coloring and concentric rings around the port."""
    
    # Find the main hub (port)
    origin_counts = df['origin_zip'].value_counts()
    dest_counts = df['destination_zip'].value_counts()
    all_zips = origin_counts.add(dest_counts, fill_value=0).sort_values(ascending=False)
    main_hub = all_zips.index[0]
    
    # Get port coordinates
    hub_data = df[df['origin_zip'] == main_hub]
    if len(hub_data) == 0:
        hub_data = df[df['destination_zip'] == main_hub]
        hub_lat, hub_lng = hub_data.iloc[0]['destination_lat'], hub_data.iloc[0]['destination_lng']
    else:
        hub_lat, hub_lng = hub_data.iloc[0]['origin_lat'], hub_data.iloc[0]['origin_lng']
    
    # Filter data by order type
    data = df[df['order_type'] == order_type].copy()
    
    # For imports, plot destinations (from port to inland)
    # For exports, plot origins (from inland to port)
    if order_type == 'import':
        data['plot_lat'] = data['destination_lat']
        data['plot_lng'] = data['destination_lng']
        data['plot_zip'] = data['destination_zip']
        title = 'Import Destinations - Colored by Rate Per Mile'
    else:
        data['plot_lat'] = data['origin_lat']
        data['plot_lng'] = data['origin_lng']
        data['plot_zip'] = data['origin_zip']
        title = 'Export Origins - Colored by Rate Per Mile'
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Add concentric rings around the port
    ring_distances_miles = [25, 50, 75, 100, 150, 200]
    miles_to_degrees = 0.014  # Approximate conversion for Southern California
    ring_colors = ['lightblue', 'lightgreen', 'yellow', 'orange', 'pink', 'lightcoral']
    
    for i, (distance, color) in enumerate(zip(ring_distances_miles, ring_colors)):
        radius_deg = distance * miles_to_degrees
        circle = Circle((hub_lng, hub_lat), radius_deg, fill=False, 
                       color=color, linewidth=2, alpha=0.7, linestyle='--')
        ax.add_patch(circle)
        
        # Add distance labels
        ax.text(hub_lng + radius_deg * 0.7, hub_lat + radius_deg * 0.7, 
                f'{distance}mi', fontsize=10, color=color, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Create scatter plot colored by RPM
    scatter = ax.scatter(data['plot_lng'], data['plot_lat'], 
                        c=data['RPM'], s=50, alpha=0.8, 
                        cmap='viridis', edgecolors='white', linewidths=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Rate Per Mile ($)', rotation=270, labelpad=20, fontsize=12)
    
    # Highlight the port with a special marker
    ax.scatter(hub_lng, hub_lat, c='red', s=400, marker='*', 
              edgecolor='black', linewidth=3, label=f'Port Hub ({main_hub})', zorder=10)
    
    # Set labels and title
    ax.set_xlabel('Longitude', fontsize=14)
    ax.set_ylabel('Latitude', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Set aspect ratio to be roughly correct for California
    ax.set_aspect('equal')
    
    # Add some statistics as text
    stats_text = f"""
    {order_type.title()} Statistics:
    Total Routes: {len(data):,}
    Avg RPM: ${data['RPM'].mean():.2f}
    RPM Range: ${data['RPM'].min():.2f} - ${data['RPM'].max():.2f}
    Avg Distance: {data['miles'].mean():.1f} miles
    """
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
            facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    return fig, ax

def create_both_maps(df):
    """Create both import and export maps."""
    
    # Create import map
    print("Creating import destinations map...")
    fig_import, ax_import = create_dot_map_with_rings(df, 'import')
    fig_import.savefig('import_destinations_rpm_map.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig_import)
    
    # Create export map
    print("Creating export origins map...")
    fig_export, ax_export = create_dot_map_with_rings(df, 'export')
    fig_export.savefig('export_origins_rpm_map.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig_export)
    
    # Create combined overview
    print("Creating combined overview...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    
    # Import subplot
    imports = df[df['order_type'] == 'import']
    hub_data = df.groupby(['origin_zip', 'destination_zip']).size().reset_index()
    origin_counts = df['origin_zip'].value_counts()
    dest_counts = df['destination_zip'].value_counts()
    all_zips = origin_counts.add(dest_counts, fill_value=0).sort_values(ascending=False)
    main_hub = all_zips.index[0]
    
    hub_data_coord = df[df['origin_zip'] == main_hub]
    if len(hub_data_coord) == 0:
        hub_data_coord = df[df['destination_zip'] == main_hub]
        hub_lat, hub_lng = hub_data_coord.iloc[0]['destination_lat'], hub_data_coord.iloc[0]['destination_lng']
    else:
        hub_lat, hub_lng = hub_data_coord.iloc[0]['origin_lat'], hub_data_coord.iloc[0]['origin_lng']
    
    # Add rings to import subplot
    ring_distances_miles = [50, 100, 150, 200]
    miles_to_degrees = 0.014
    ring_colors = ['lightblue', 'lightgreen', 'yellow', 'orange']
    
    for distance, color in zip(ring_distances_miles, ring_colors):
        radius_deg = distance * miles_to_degrees
        circle = Circle((hub_lng, hub_lat), radius_deg, fill=False, 
                       color=color, linewidth=2, alpha=0.5, linestyle='--')
        ax1.add_patch(circle)
    
    scatter1 = ax1.scatter(imports['destination_lng'], imports['destination_lat'], 
                          c=imports['RPM'], s=30, alpha=0.8, 
                          cmap='viridis', edgecolors='white', linewidths=0.3)
    ax1.scatter(hub_lng, hub_lat, c='red', s=300, marker='*', 
               edgecolor='black', linewidth=2, zorder=10)
    ax1.set_title('Import Destinations', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Export subplot
    exports = df[df['order_type'] == 'export']
    
    # Add rings to export subplot
    for distance, color in zip(ring_distances_miles, ring_colors):
        radius_deg = distance * miles_to_degrees
        circle = Circle((hub_lng, hub_lat), radius_deg, fill=False, 
                       color=color, linewidth=2, alpha=0.5, linestyle='--')
        ax2.add_patch(circle)
    
    scatter2 = ax2.scatter(exports['origin_lng'], exports['origin_lat'], 
                          c=exports['RPM'], s=30, alpha=0.8, 
                          cmap='plasma', edgecolors='white', linewidths=0.3)
    ax2.scatter(hub_lng, hub_lat, c='red', s=300, marker='*', 
               edgecolor='black', linewidth=2, zorder=10)
    ax2.set_title('Export Origins', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # Add colorbars
    cbar1 = plt.colorbar(scatter1, ax=ax1, shrink=0.8)
    cbar1.set_label('RPM ($)', rotation=270, labelpad=15)
    cbar2 = plt.colorbar(scatter2, ax=ax2, shrink=0.8)
    cbar2.set_label('RPM ($)', rotation=270, labelpad=15)
    
    plt.suptitle('Port Drayage Network - Dots Colored by Rate Per Mile', 
                 fontsize=18, fontweight='bold', y=0.95)
    plt.tight_layout()
    fig.savefig('combined_import_export_rpm_map.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def main():
    """Main function to create the dot maps."""
    print("Creating Port Drayage Dot Maps with RPM Coloring...")
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Create all maps
    create_both_maps(df)
    
    print("\nDot maps created successfully!")
    print("Files generated:")
    print("- import_destinations_rpm_map.png (Import destinations with concentric rings)")
    print("- export_origins_rpm_map.png (Export origins with concentric rings)")
    print("- combined_import_export_rpm_map.png (Side-by-side comparison)")
    
    # Print summary
    print(f"\nData Summary:")
    print(f"Total routes: {len(df):,}")
    print(f"Import routes: {len(df[df['order_type'] == 'import']):,}")
    print(f"Export routes: {len(df[df['order_type'] == 'export']):,}")
    print(f"RPM range: ${df['RPM'].min():.2f} - ${df['RPM'].max():.2f}")
    print(f"Average RPM: ${df['RPM'].mean():.2f}")

if __name__ == "__main__":
    main()
