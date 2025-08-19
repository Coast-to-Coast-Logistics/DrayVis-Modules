# DrayVis Professional GUI Rate Estimator

A modern, user-friendly graphical interface for the DrayVis intelligent rate estimation system. No more terminal commands - everything you need in a clean, professional desktop application!

## 🚀 Quick Start

### Windows Users
1. **Double-click `DrayVis_GUI.bat`** - The easiest way to launch the application

### All Platforms
2. **Run the launcher**: `python launch_gui.py`
3. **Direct launch**: `python gui_rate_estimator.py`

## ✨ Features

### 🎯 Single Rate Estimation
- Enter any zip code and get instant rate estimates
- Real-time confidence levels and explanations
- Detailed analysis including distance to port and rate ranges

### 📊 Batch Processing
- Estimate rates for multiple zip codes at once
- Load zip codes from text or CSV files
- Progress tracking and error reporting
- Success rate analytics

### 📈 Professional Results Display
- Sortable table with all key metrics
- Detailed explanations for each estimate
- Confidence distribution analysis
- Export capabilities (CSV/JSON)

### 🔧 Advanced Options
- Verbose output toggle
- Auto-export functionality
- Clear results management
- Professional status tracking

## 🖥️ Interface Overview

### Main Sections

1. **Input Panel** (Left)
   - Single zip code estimation
   - Batch processing text area
   - File loading capabilities
   - Configuration options

2. **Results Panel** (Right)
   - Comprehensive results table
   - Export controls
   - Detailed explanations pane
   - Visual confidence indicators

3. **Status Bar** (Bottom)
   - Real-time progress tracking
   - System status indicators
   - Background processing alerts

## 📋 Using the GUI

### Single Zip Code Estimation
1. Enter a zip code in the "Single Zip Code" field
2. Click "Estimate Rate" or press Enter
3. View results in the results table
4. Click on any result to see detailed explanation

### Batch Processing
1. Enter multiple zip codes in the batch text area (comma or line separated)
2. Or click "Load from File" to import from a text/CSV file
3. Click "Batch Estimate" to process all zip codes
4. Monitor progress in the status bar
5. Review completion summary when finished

### Exporting Results
- **CSV Export**: Full data with all metrics and explanations
- **JSON Export**: Structured data with timestamps
- **Auto-Export**: Automatically save results after each batch

### File Import Formats
The system accepts zip codes in various formats:
- Comma-separated: `90210, 91106, 92660`
- Line-separated: One zip code per line
- Mixed with text: Automatically extracts 5-digit zip codes
- CSV files: Reads from any column containing zip codes

## 🎯 Understanding Results

### Results Table Columns
- **Zip Code**: Target zip code
- **RPM**: Estimated rate per mile
- **Confidence**: Confidence level and category
- **Distance**: Distance to LA Port in miles
- **Method**: Estimation method used
- **Range**: Rate range (low - high)

### Confidence Categories
- **Very High (80%+)**: Based on nearby historical data
- **High (60-79%)**: Strong neighbor consensus
- **Medium (40-59%)**: Moderate data availability
- **Low (<40%)**: Limited data, general patterns used

### Estimation Methods
- **Historical Data**: Direct historical rate for this zip code
- **K-Nearest Neighbors**: Rate based on similar nearby locations
- **Distance-based ML**: Machine learning model using distance patterns
- **Regional Average**: Fallback regional estimation

## ⚙️ Configuration

The GUI automatically uses the optimized configuration from the intelligent rate estimator:

- **Time Weighting**: Recent data weighted more heavily
- **Port Distance Similarity**: Neighbors with similar port distances prioritized
- **Geographic Proximity**: Closer zip codes get higher weight
- **Sample Size Consideration**: More data points increase confidence

## 🔧 Troubleshooting

### Application Won't Start
1. Ensure Python is installed and in your PATH
2. Run `python test_gui.py` to check dependencies
3. Install missing packages: `pip install pandas numpy geopy scikit-learn`

### Estimation Errors
- **Invalid Zip Code**: Enter valid 5-digit US zip codes
- **No Data Found**: Try nearby zip codes or different regions
- **Loading Issues**: Restart the application

### File Import Issues
- Ensure files contain valid 5-digit zip codes
- Use standard text encoding (UTF-8)
- Remove special characters or formatting

## 📊 Export File Formats

### CSV Export
```csv
zip_code,estimated_rpm,confidence_level,confidence_category,method_used,distance_to_port,rate_range_low,rate_range_high,nearest_neighbors,explanation
90210,13.74,96.5,Very High,K-Nearest Neighbors,27.2,12.11,15.37,"90069,90210,90211",High confidence estimate based on nearby zip codes...
```

### JSON Export
```json
[
  {
    "zip_code": "90210",
    "estimated_rpm": 13.74,
    "confidence_level": 96.5,
    "confidence_category": "Very High",
    "method_used": "K-Nearest Neighbors",
    "distance_to_port": 27.2,
    "rate_range": [12.11, 15.37],
    "nearest_neighbors": ["90069", "90210", "90211"],
    "explanation": "High confidence estimate...",
    "timestamp": "2025-08-19T10:30:45"
  }
]
```

## 🎨 Interface Tips

- **Keyboard Shortcuts**: Press Enter in the single zip field to estimate
- **Selection**: Click any result row to see detailed explanation
- **Sorting**: Click column headers to sort results
- **Progress**: Watch the status bar for real-time updates
- **Clear**: Use "Clear Results" to start fresh

## 🔮 Advanced Features

### Background Processing
- All estimation runs in background threads
- GUI remains responsive during batch processing
- Real-time progress updates

### Error Handling
- Graceful error reporting
- Partial batch processing (continues on errors)
- Detailed error messages in completion summary

### Memory Management
- Efficient data handling for large batches
- Automatic cleanup of temporary data
- Optimized for long-running sessions

## 💡 Best Practices

1. **Start Small**: Test with a few zip codes before large batches
2. **Check Results**: Review confidence levels and explanations
3. **Export Early**: Save results frequently for important analyses
4. **Use Verbose**: Enable verbose output for detailed insights
5. **Monitor Progress**: Watch status bar during batch processing

## 🔗 Integration

The GUI seamlessly integrates with the enhanced DrayVis rate estimator, providing:
- All Version 2.1 enhancements (time weighting, port distance similarity)
- Professional confidence scoring
- Multiple estimation methods
- Real-time data processing

---

**Ready to start?** Double-click `DrayVis_GUI.bat` or run `python launch_gui.py`

For technical support or questions about the rate estimation algorithms, refer to the `intelligent_rate_estimator.py` documentation.
