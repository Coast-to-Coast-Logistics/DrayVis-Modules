#!/usr/bin/env python3
"""
Interactive Port Drayage Visualization with Toggle
Creates an interactive visualization with dots colored by RPM and concentric rings.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

def load_data():
    """Load the port drayage CSV data."""
    try:
        df = pd.read_csv('data/port_drayage_dummy_data.csv')
        df['date'] = pd.to_datetime(df['date'])
        return df
    except FileNotFoundError:
        print("Error: Could not find data/port_drayage_dummy_data.csv")
        return None

def create_interactive_toggle_map(df):
    """Create an interactive map with import/export toggle and RPM coloring."""
    
    # Find the main hub (port)
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
    
    # Separate import and export data
    imports = df[df['order_type'] == 'import'].copy()
    exports = df[df['order_type'] == 'export'].copy()
    
    # For imports, we show destinations (from port to inland)
    imports['plot_lat'] = imports['destination_lat']
    imports['plot_lng'] = imports['destination_lng']
    imports['plot_zip'] = imports['destination_zip']
    
    # For exports, we show origins (from inland to port)
    exports['plot_lat'] = exports['origin_lat']
    exports['plot_lng'] = exports['origin_lng']
    exports['plot_zip'] = exports['origin_zip']
    
    # Create the figure
    fig = go.Figure()
    
    # Add concentric rings around the port
    ring_distances_miles = [25, 50, 75, 100, 150, 200]
    miles_to_degrees = 0.014  # Approximate conversion for Southern California
    
    for distance in ring_distances_miles:
        radius_deg = distance * miles_to_degrees
        
        # Create circle points
        theta = np.linspace(0, 2*np.pi, 100)
        circle_lat = hub_lat + radius_deg * np.sin(theta)
        circle_lng = hub_lng + radius_deg * np.cos(theta)
        
        fig.add_trace(go.Scattermapbox(
            lat=circle_lat,
            lon=circle_lng,
            mode='lines',
            line=dict(width=2, color='gray'),
            opacity=0.5,
            name=f'{distance} miles',
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Add distance label
        fig.add_trace(go.Scattermapbox(
            lat=[hub_lat + radius_deg * 0.7],
            lon=[hub_lng + radius_deg * 0.7],
            mode='text',
            text=[f'{distance}mi'],
            textfont=dict(size=10, color='gray'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add import destinations
    fig.add_trace(go.Scattermapbox(
        lat=imports['plot_lat'],
        lon=imports['plot_lng'],
        mode='markers',
        marker=dict(
            size=8,
            color=imports['RPM'],
            colorscale='Viridis',
            cmin=df['RPM'].min(),
            cmax=df['RPM'].max(),
            colorbar=dict(
                title=dict(text="Rate Per Mile ($)")
            ),
            line=dict(width=1, color='white')
        ),
        text=[f"Zip: {row['plot_zip']}<br>Carrier: {row['carrier']}<br>Rate: ${row['rate']:.2f}<br>Distance: {row['miles']:.1f}mi<br>RPM: ${row['RPM']:.2f}" 
              for _, row in imports.iterrows()],
        hovertemplate='<b>Import Destination</b><br>%{text}<extra></extra>',
        name='Import Destinations',
        visible=True
    ))
    
    # Add export origins
    fig.add_trace(go.Scattermapbox(
        lat=exports['plot_lat'],
        lon=exports['plot_lng'],
        mode='markers',
        marker=dict(
            size=8,
            color=exports['RPM'],
            colorscale='Plasma',
            cmin=df['RPM'].min(),
            cmax=df['RPM'].max(),
            showscale=False,  # Hide colorbar for exports to avoid duplication
            line=dict(width=1, color='white')
        ),
        text=[f"Zip: {row['plot_zip']}<br>Carrier: {row['carrier']}<br>Rate: ${row['rate']:.2f}<br>Distance: {row['miles']:.1f}mi<br>RPM: ${row['RPM']:.2f}" 
              for _, row in exports.iterrows()],
        hovertemplate='<b>Export Origin</b><br>%{text}<extra></extra>',
        name='Export Origins',
        visible=False
    ))
    
    # Add port marker
    fig.add_trace(go.Scattermapbox(
        lat=[hub_lat],
        lon=[hub_lng],
        mode='markers',
        marker=dict(
            size=20,
            color='red',
            symbol='star'
        ),
        text=[f'Port Hub: {main_hub}'],
        hovertemplate='<b>%{text}</b><extra></extra>',
        name='Port Hub',
        showlegend=True
    ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Port Drayage Network - Interactive Toggle View',
            'x': 0.5,
            'font': {'size': 20}
        },
        mapbox=dict(
            style='open-street-map',
            center=dict(lat=hub_lat, lon=hub_lng),
            zoom=8
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=700,
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(
                        args=[{"visible": [True] * (len(ring_distances_miles) * 2) + [True, False, True]}],
                        label="Import Destinations",
                        method="update"
                    ),
                    dict(
                        args=[{"visible": [True] * (len(ring_distances_miles) * 2) + [False, True, True]}],
                        label="Export Origins",
                        method="update"
                    ),
                    dict(
                        args=[{"visible": [True] * (len(ring_distances_miles) * 2) + [True, True, True]}],
                        label="Both",
                        method="update"
                    )
                ]),
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.01,
                xanchor="left",
                y=1.0,
                yanchor="top"
            ),
        ]
    )
    
    return fig

def create_rpm_analysis_dashboard(df):
    """Create a comprehensive RPM analysis dashboard."""
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('RPM Distribution by Order Type', 'RPM vs Distance', 
                       'RPM by Top Carriers', 'Monthly RPM Trends'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 1. RPM Distribution
    for i, order_type in enumerate(['import', 'export']):
        data = df[df['order_type'] == order_type]['RPM']
        fig.add_trace(
            go.Histogram(
                x=data,
                name=f'{order_type.title()} RPM',
                opacity=0.7,
                nbinsx=30
            ),
            row=1, col=1
        )
    
    # 2. RPM vs Distance scatter
    fig.add_trace(
        go.Scatter(
            x=df[df['order_type'] == 'import']['miles'],
            y=df[df['order_type'] == 'import']['RPM'],
            mode='markers',
            name='Import',
            marker=dict(color='blue', size=4, opacity=0.6)
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=df[df['order_type'] == 'export']['miles'],
            y=df[df['order_type'] == 'export']['RPM'],
            mode='markers',
            name='Export',
            marker=dict(color='red', size=4, opacity=0.6)
        ),
        row=1, col=2
    )
    
    # 3. RPM by Top Carriers
    top_carriers = df.groupby('carrier')['RPM'].mean().nlargest(10)
    fig.add_trace(
        go.Bar(
            x=top_carriers.values,
            y=[name[:20] + '...' if len(name) > 20 else name for name in top_carriers.index],
            orientation='h',
            name='Avg RPM',
            marker_color='steelblue'
        ),
        row=2, col=1
    )
    
    # 4. Monthly RPM trends
    df_monthly = df.copy()
    df_monthly['month'] = df_monthly['date'].dt.to_period('M').astype(str)
    monthly_rpm = df_monthly.groupby(['month', 'order_type'])['RPM'].mean().reset_index()
    
    for order_type in ['import', 'export']:
        data = monthly_rpm[monthly_rpm['order_type'] == order_type]
        fig.add_trace(
            go.Scatter(
                x=data['month'],
                y=data['RPM'],
                mode='lines+markers',
                name=f'{order_type.title()} RPM Trend',
                line=dict(width=3)
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Rate Per Mile (RPM) Analysis Dashboard',
            'x': 0.5,
            'font': {'size': 20}
        },
        height=800,
        showlegend=True
    )
    
    # Update axis labels
    fig.update_xaxes(title_text="RPM ($)", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_xaxes(title_text="Distance (miles)", row=1, col=2)
    fig.update_yaxes(title_text="RPM ($)", row=1, col=2)
    fig.update_xaxes(title_text="Average RPM ($)", row=2, col=1)
    fig.update_yaxes(title_text="Carrier", row=2, col=1)
    fig.update_xaxes(title_text="Month", row=2, col=2)
    fig.update_yaxes(title_text="Average RPM ($)", row=2, col=2)
    
    return fig

def main():
    """Main function to create interactive visualizations."""
    print("Creating Interactive Port Drayage Visualizations...")
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    print("Creating interactive toggle map...")
    map_fig = create_interactive_toggle_map(df)
    map_fig.write_html('interactive_toggle_map.html')
    print("Interactive map saved as: interactive_toggle_map.html")
    
    print("Creating RPM analysis dashboard...")
    rpm_fig = create_rpm_analysis_dashboard(df)
    rpm_fig.write_html('rpm_analysis_dashboard.html')
    print("RPM dashboard saved as: rpm_analysis_dashboard.html")
    
    print("\nInteractive visualizations complete!")
    print("Files created:")
    print("- interactive_toggle_map.html (dots colored by RPM with toggle)")
    print("- rpm_analysis_dashboard.html (comprehensive RPM analysis)")
    
    # Print summary statistics
    print(f"\nRPM Statistics:")
    print(f"Average RPM: ${df['RPM'].mean():.2f}")
    print(f"Import Average RPM: ${df[df['order_type'] == 'import']['RPM'].mean():.2f}")
    print(f"Export Average RPM: ${df[df['order_type'] == 'export']['RPM'].mean():.2f}")
    print(f"RPM Range: ${df['RPM'].min():.2f} - ${df['RPM'].max():.2f}")

if __name__ == "__main__":
    main()
