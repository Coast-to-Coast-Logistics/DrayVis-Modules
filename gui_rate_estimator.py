"""
DrayVis GUI Rate Estimator - Professional Desktop Interface
===========================================================

Modern graphical interface for the intelligent rate estimator.
Provides intuitive controls, real-time results, and professional visualization.

Author: DrayVis Analytics Team  
Date: August 19, 2025
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import pandas as pd
import json
from datetime import datetime
from typing import List, Dict
import threading
import queue
from intelligent_rate_estimator import IntelligentRateEstimator, RateEstimate

class DrayVisGUI:
    """Professional GUI for DrayVis Rate Estimator"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("DrayVis Rate Estimator")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize estimator
        self.estimator = None
        self.estimation_history = []
        self.current_results = []
        
        # Setup styles
        self.setup_styles()
        
        # Create main interface
        self.create_widgets()
        
        # Initialize estimator in background
        self.init_estimator_async()
        
    def setup_styles(self):
        """Configure custom styles for professional appearance"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure custom styles
        style.configure('Header.TLabel', 
                       font=('Arial', 14, 'bold'),
                       background='#f0f0f0',
                       foreground='#2c3e50')
        
        style.configure('Subheader.TLabel',
                       font=('Arial', 11, 'bold'),
                       background='#f0f0f0',
                       foreground='#34495e')
        
        style.configure('Action.TButton',
                       font=('Arial', 10, 'bold'),
                       foreground='white')
        
        style.map('Action.TButton',
                 background=[('active', '#2980b9'), ('!active', '#3498db')])
        
        style.configure('Success.TLabel',
                       font=('Arial', 10),
                       background='#f0f0f0',
                       foreground='#27ae60')
        
        style.configure('Error.TLabel',
                       font=('Arial', 10),
                       background='#f0f0f0',
                       foreground='#e74c3c')
    
    def create_widgets(self):
        """Create and arrange all GUI widgets"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Header
        self.create_header(main_frame)
        
        # Input section
        self.create_input_section(main_frame)
        
        # Results section
        self.create_results_section(main_frame)
        
        # Status bar
        self.create_status_bar(main_frame)
    
    def create_header(self, parent):
        """Create header section"""
        header_frame = ttk.Frame(parent)
        header_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
        # Title
        title_label = ttk.Label(header_frame, 
                               text="🚛 DrayVis Rate Estimator",
                               style='Header.TLabel')
        title_label.grid(row=0, column=0, sticky=tk.W)
        
        # Status indicator
        self.status_label = ttk.Label(header_frame, 
                                     text="⏳ Initializing...",
                                     font=('Arial', 10))
        self.status_label.grid(row=0, column=1, sticky=tk.E)
        
        header_frame.columnconfigure(1, weight=1)
    
    def create_input_section(self, parent):
        """Create input controls section"""
        input_frame = ttk.LabelFrame(parent, text="Rate Estimation", padding="15")
        input_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N), padx=(0, 10))
        
        # Single zip estimation
        ttk.Label(input_frame, text="Single Zip Code:", style='Subheader.TLabel').grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        zip_frame = ttk.Frame(input_frame)
        zip_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        zip_frame.columnconfigure(0, weight=1)
        
        self.single_zip_var = tk.StringVar()
        self.single_zip_entry = ttk.Entry(zip_frame, textvariable=self.single_zip_var, font=('Arial', 11))
        self.single_zip_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        self.single_zip_entry.bind('<Return>', lambda e: self.estimate_single_zip())
        
        self.estimate_btn = ttk.Button(zip_frame, text="Estimate Rate", 
                                      command=self.estimate_single_zip,
                                      style='Action.TButton')
        self.estimate_btn.grid(row=0, column=1)
        self.estimate_btn.configure(state='disabled')
        
        # Batch estimation
        ttk.Label(input_frame, text="Batch Estimation:", style='Subheader.TLabel').grid(row=2, column=0, sticky=tk.W, pady=(10, 5))
        
        batch_frame = ttk.Frame(input_frame)
        batch_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        batch_frame.columnconfigure(0, weight=1)
        
        self.batch_text = tk.Text(batch_frame, height=3, width=30, font=('Arial', 10))
        self.batch_text.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        
        batch_btn_frame = ttk.Frame(batch_frame)
        batch_btn_frame.grid(row=0, column=1, sticky=tk.N)
        
        self.batch_btn = ttk.Button(batch_btn_frame, text="Batch Estimate",
                                   command=self.estimate_batch,
                                   style='Action.TButton')
        self.batch_btn.grid(row=0, column=0, pady=(0, 5))
        self.batch_btn.configure(state='disabled')
        
        ttk.Button(batch_btn_frame, text="Load from File",
                  command=self.load_zip_file).grid(row=1, column=0)
        
        # Options
        options_frame = ttk.LabelFrame(input_frame, text="Options", padding="10")
        options_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.verbose_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Verbose output", 
                       variable=self.verbose_var).grid(row=0, column=0, sticky=tk.W)
        
        self.auto_export_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Auto-export results", 
                       variable=self.auto_export_var).grid(row=0, column=1, sticky=tk.W, padx=(20, 0))
    
    def create_results_section(self, parent):
        """Create results display section"""
        results_frame = ttk.LabelFrame(parent, text="Results", padding="15")
        results_frame.grid(row=1, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(1, weight=1)
        
        # Results toolbar
        toolbar_frame = ttk.Frame(results_frame)
        toolbar_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        toolbar_frame.columnconfigure(0, weight=1)
        
        # Results count
        self.results_count_label = ttk.Label(toolbar_frame, text="No results yet")
        self.results_count_label.grid(row=0, column=0, sticky=tk.W)
        
        # Export buttons
        export_frame = ttk.Frame(toolbar_frame)
        export_frame.grid(row=0, column=1, sticky=tk.E)
        
        ttk.Button(export_frame, text="Export CSV",
                  command=self.export_csv).grid(row=0, column=0, padx=(0, 5))
        
        ttk.Button(export_frame, text="Export JSON",
                  command=self.export_json).grid(row=0, column=1, padx=(0, 5))
        
        ttk.Button(export_frame, text="Clear Results",
                  command=self.clear_results).grid(row=0, column=2)
        
        # Results display
        self.create_results_display(results_frame)
    
    def create_results_display(self, parent):
        """Create scrollable results display"""
        # Create treeview for results table
        columns = ('Zip Code', 'RPM', 'Confidence', 'Distance', 'Method', 'Range')
        self.results_tree = ttk.Treeview(parent, columns=columns, show='headings', height=15)
        
        # Configure columns
        column_widths = {'Zip Code': 80, 'RPM': 80, 'Confidence': 90, 
                        'Distance': 80, 'Method': 120, 'Range': 120}
        
        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=column_widths.get(col, 100), anchor=tk.CENTER)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.results_tree.yview)
        h_scrollbar = ttk.Scrollbar(parent, orient=tk.HORIZONTAL, command=self.results_tree.xview)
        self.results_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Grid layout
        self.results_tree.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        v_scrollbar.grid(row=1, column=1, sticky=(tk.N, tk.S))
        h_scrollbar.grid(row=2, column=0, sticky=(tk.W, tk.E))
        
        # Bind selection event
        self.results_tree.bind('<<TreeviewSelect>>', self.on_result_select)
        
        # Details panel
        self.details_frame = ttk.LabelFrame(parent, text="Details", padding="10")
        self.details_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.details_text = scrolledtext.ScrolledText(self.details_frame, height=6, 
                                                     font=('Consolas', 9), wrap=tk.WORD)
        self.details_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        self.details_frame.columnconfigure(0, weight=1)
    
    def create_status_bar(self, parent):
        """Create status bar"""
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        status_frame.columnconfigure(1, weight=1)
        
        self.progress_var = tk.StringVar(value="Ready")
        self.progress_label = ttk.Label(status_frame, textvariable=self.progress_var)
        self.progress_label.grid(row=0, column=0, sticky=tk.W)
        
        self.progress_bar = ttk.Progressbar(status_frame, mode='indeterminate')
        self.progress_bar.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0))
    
    def init_estimator_async(self):
        """Initialize estimator in background thread"""
        def init_worker():
            try:
                self.estimator = IntelligentRateEstimator()
                self.root.after(0, self.on_estimator_ready)
            except Exception as e:
                self.root.after(0, lambda: self.on_estimator_error(str(e)))
        
        self.progress_var.set("Loading rate estimator...")
        self.progress_bar.start()
        
        thread = threading.Thread(target=init_worker, daemon=True)
        thread.start()
    
    def on_estimator_ready(self):
        """Called when estimator is ready"""
        self.progress_bar.stop()
        self.progress_var.set("Ready")
        self.status_label.configure(text="✅ Ready", foreground='#27ae60')
        
        # Enable controls
        self.estimate_btn.configure(state='normal')
        self.batch_btn.configure(state='normal')
        
        messagebox.showinfo("DrayVis", "Rate estimator loaded successfully!")
    
    def on_estimator_error(self, error_msg):
        """Called when estimator fails to load"""
        self.progress_bar.stop()
        self.progress_var.set("Error")
        self.status_label.configure(text="❌ Error", foreground='#e74c3c')
        
        messagebox.showerror("Error", f"Failed to load rate estimator:\n{error_msg}")
    
    def estimate_single_zip(self):
        """Estimate rate for single zip code"""
        zip_code = self.single_zip_var.get().strip()
        if not zip_code:
            messagebox.showwarning("Input Required", "Please enter a zip code.")
            return
        
        if not self.estimator:
            messagebox.showerror("Error", "Rate estimator not ready.")
            return
        
        def estimate_worker():
            try:
                result = self.estimator.estimate_rate(zip_code, verbose=self.verbose_var.get())
                self.root.after(0, lambda: self.on_estimate_complete([result]))
            except Exception as e:
                self.root.after(0, lambda: self.on_estimate_error(str(e)))
        
        self.start_estimation(f"Estimating rate for {zip_code}...")
        
        thread = threading.Thread(target=estimate_worker, daemon=True)
        thread.start()
    
    def estimate_batch(self):
        """Estimate rates for multiple zip codes"""
        batch_text = self.batch_text.get("1.0", tk.END).strip()
        if not batch_text:
            messagebox.showwarning("Input Required", "Please enter zip codes for batch estimation.")
            return
        
        # Parse zip codes
        zip_codes = []
        for line in batch_text.split('\n'):
            for zip_code in line.split(','):
                zip_code = zip_code.strip()
                if zip_code:
                    zip_codes.append(zip_code)
        
        if not zip_codes:
            messagebox.showwarning("Input Required", "No valid zip codes found.")
            return
        
        if not self.estimator:
            messagebox.showerror("Error", "Rate estimator not ready.")
            return
        
        def estimate_worker():
            results = []
            errors = []
            
            for i, zip_code in enumerate(zip_codes):
                try:
                    result = self.estimator.estimate_rate(zip_code, verbose=False)
                    results.append(result)
                    
                    # Update progress
                    progress_msg = f"Processing {i+1}/{len(zip_codes)}: {zip_code}"
                    self.root.after(0, lambda msg=progress_msg: self.progress_var.set(msg))
                    
                except Exception as e:
                    errors.append({'zip_code': zip_code, 'error': str(e)})
            
            self.root.after(0, lambda: self.on_batch_complete(results, errors))
        
        self.start_estimation(f"Processing {len(zip_codes)} zip codes...")
        
        thread = threading.Thread(target=estimate_worker, daemon=True)
        thread.start()
    
    def start_estimation(self, message):
        """Start estimation process"""
        self.progress_var.set(message)
        self.progress_bar.start()
        self.estimate_btn.configure(state='disabled')
        self.batch_btn.configure(state='disabled')
    
    def on_estimate_complete(self, results):
        """Handle single estimation completion"""
        self.progress_bar.stop()
        self.progress_var.set("Ready")
        self.estimate_btn.configure(state='normal')
        self.batch_btn.configure(state='normal')
        
        self.add_results(results)
        self.single_zip_var.set("")  # Clear input
        
        if self.auto_export_var.get() and results:
            self.export_csv()
    
    def on_batch_complete(self, results, errors):
        """Handle batch estimation completion"""
        self.progress_bar.stop()
        self.progress_var.set("Ready")
        self.estimate_btn.configure(state='normal')
        self.batch_btn.configure(state='normal')
        
        self.add_results(results)
        
        # Show summary
        success_rate = len(results) / (len(results) + len(errors)) * 100 if (results or errors) else 0
        message = f"Batch estimation complete!\n\nSuccessful: {len(results)}\nErrors: {len(errors)}\nSuccess Rate: {success_rate:.1f}%"
        
        if errors:
            error_details = "\n".join([f"• {e['zip_code']}: {e['error']}" for e in errors[:5]])
            if len(errors) > 5:
                error_details += f"\n... and {len(errors) - 5} more errors"
            message += f"\n\nFirst few errors:\n{error_details}"
        
        messagebox.showinfo("Batch Complete", message)
        
        if self.auto_export_var.get() and results:
            self.export_csv()
    
    def on_estimate_error(self, error_msg):
        """Handle estimation error"""
        self.progress_bar.stop()
        self.progress_var.set("Ready")
        self.estimate_btn.configure(state='normal')
        self.batch_btn.configure(state='normal')
        
        messagebox.showerror("Estimation Error", error_msg)
    
    def add_results(self, results):
        """Add results to the display"""
        for result in results:
            # Add to tree
            confidence_text = f"{result.confidence_level:.1f}% ({result.confidence_category})"
            range_text = f"${result.rate_range[0]:.2f} - ${result.rate_range[1]:.2f}"
            
            self.results_tree.insert('', 'end', values=(
                result.zip_code,
                f"${result.estimated_rpm:.2f}",
                confidence_text,
                f"{result.distance_to_port:.1f} mi",
                result.method_used,
                range_text
            ))
            
            # Add to current results
            self.current_results.append(result)
            self.estimation_history.append({
                'timestamp': datetime.now().isoformat(),
                'result': result,
                'success': True
            })
        
        # Update count
        self.results_count_label.configure(text=f"{len(self.current_results)} results")
    
    def on_result_select(self, event):
        """Handle result selection"""
        selection = self.results_tree.selection()
        if not selection:
            return
        
        # Get selected item
        item = self.results_tree.item(selection[0])
        zip_code = item['values'][0]
        
        # Find corresponding result
        result = next((r for r in self.current_results if r.zip_code == zip_code), None)
        if not result:
            return
        
        # Show details
        details = f"""Zip Code: {result.zip_code}
Estimated RPM: ${result.estimated_rpm:.2f}
Confidence: {result.confidence_level:.1f}% ({result.confidence_category})
Distance to Port: {result.distance_to_port:.1f} miles
Method Used: {result.method_used}
Rate Range: ${result.rate_range[0]:.2f} - ${result.rate_range[1]:.2f}
Nearest Neighbors: {', '.join(result.nearest_neighbors)}

Explanation:
{result.explanation}"""
        
        self.details_text.delete('1.0', tk.END)
        self.details_text.insert('1.0', details)
    
    def load_zip_file(self):
        """Load zip codes from file"""
        filename = filedialog.askopenfilename(
            title="Load Zip Codes",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            with open(filename, 'r') as f:
                content = f.read()
            
            # Parse content
            zip_codes = []
            for line in content.split('\n'):
                for item in line.split(','):
                    item = item.strip().strip('"').strip("'")
                    if item and item.isdigit() and len(item) == 5:
                        zip_codes.append(item)
            
            if zip_codes:
                self.batch_text.delete('1.0', tk.END)
                self.batch_text.insert('1.0', ', '.join(zip_codes))
                messagebox.showinfo("Success", f"Loaded {len(zip_codes)} zip codes from file.")
            else:
                messagebox.showwarning("No Data", "No valid zip codes found in file.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")
    
    def export_csv(self):
        """Export results to CSV"""
        if not self.current_results:
            messagebox.showwarning("No Data", "No results to export.")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Export to CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            # Prepare export data
            export_data = []
            for result in self.current_results:
                export_data.append({
                    'zip_code': result.zip_code,
                    'estimated_rpm': result.estimated_rpm,
                    'confidence_level': result.confidence_level,
                    'confidence_category': result.confidence_category,
                    'method_used': result.method_used,
                    'distance_to_port': result.distance_to_port,
                    'rate_range_low': result.rate_range[0],
                    'rate_range_high': result.rate_range[1],
                    'nearest_neighbors': ','.join(result.nearest_neighbors),
                    'explanation': result.explanation
                })
            
            df = pd.DataFrame(export_data)
            df.to_csv(filename, index=False)
            
            messagebox.showinfo("Export Complete", f"Results exported to:\n{filename}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export CSV:\n{str(e)}")
    
    def export_json(self):
        """Export results to JSON"""
        if not self.current_results:
            messagebox.showwarning("No Data", "No results to export.")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Export to JSON",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            # Prepare export data
            export_data = []
            for result in self.current_results:
                export_data.append({
                    'zip_code': result.zip_code,
                    'estimated_rpm': result.estimated_rpm,
                    'confidence_level': result.confidence_level,
                    'confidence_category': result.confidence_category,
                    'method_used': result.method_used,
                    'distance_to_port': result.distance_to_port,
                    'rate_range': result.rate_range,
                    'nearest_neighbors': result.nearest_neighbors,
                    'explanation': result.explanation,
                    'timestamp': datetime.now().isoformat()
                })
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            messagebox.showinfo("Export Complete", f"Results exported to:\n{filename}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export JSON:\n{str(e)}")
    
    def clear_results(self):
        """Clear all results"""
        if not self.current_results:
            return
        
        if messagebox.askyesno("Clear Results", "Are you sure you want to clear all results?"):
            # Clear tree
            for item in self.results_tree.get_children():
                self.results_tree.delete(item)
            
            # Clear data
            self.current_results.clear()
            
            # Clear details
            self.details_text.delete('1.0', tk.END)
            
            # Update count
            self.results_count_label.configure(text="No results")

def main():
    """Main function"""
    root = tk.Tk()
    app = DrayVisGUI(root)
    
    # Handle window closing
    def on_closing():
        if messagebox.askokcancel("Quit", "Do you want to quit DrayVis?"):
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Start the GUI
    root.mainloop()

if __name__ == "__main__":
    main()
