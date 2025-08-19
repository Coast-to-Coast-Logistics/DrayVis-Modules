"""
Comprehensive Validation Orchestrator
=====================================

Advanced validation system that:
1. Runs all configurations in parallel
2. Generates performance comparison reports
3. Creates visualization charts
4. Outputs structured data for AI analysis
5. Enables automatic system optimization

This builds on the FastKFoldValidator to provide comprehensive insights
and auto-optimization capabilities.
"""

import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Import our existing optimized validator
from kfold_validation import FastKFoldValidator

class ComprehensiveValidationOrchestrator:
    """
    Orchestrates comprehensive validation across all configurations
    with performance analysis and auto-optimization capabilities
    """
    
    def __init__(self, output_dir: str = "reports"):
        """Initialize the comprehensive validation orchestrator"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "charts").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        
        self.validator = FastKFoldValidator()
        self.results_history = []
        self.optimization_recommendations = []
        
        print("🚀 Comprehensive Validation Orchestrator Initialized")
        print(f"📁 Output directory: {self.output_dir}")
        
    def run_comprehensive_analysis(self, fast_mode: bool = True) -> Dict[str, Any]:
        """
        Run comprehensive validation analysis across all configurations
        
        Args:
            fast_mode: If True, use quick validation for speed
            
        Returns:
            Dictionary containing all analysis results
        """
        print("\n🔬 COMPREHENSIVE VALIDATION ANALYSIS")
        print("=" * 60)
        
        start_time = time.time()
        
        # Get all available configurations
        configurations = self.validator.config_tester.get_configuration_names()
        print(f"📊 Testing {len(configurations)} configurations:")
        for i, config in enumerate(configurations, 1):
            print(f"   {i}. {config}")
        
        print(f"\n⚡ Mode: {'Fast validation' if fast_mode else 'Full validation'}")
        print("🚀 Starting parallel validation...\n")
        
        # Run all configurations in parallel
        all_results = self._run_parallel_validation(configurations, fast_mode)
        
        # Generate comprehensive analysis
        analysis_results = self._generate_comprehensive_analysis(all_results)
        
        # Create performance dashboard
        self._create_performance_dashboard(analysis_results)
        
        # Generate structured output for AI analysis
        self._generate_ai_readable_output(analysis_results)
        
        # Run auto-optimization analysis
        optimization_results = self._run_auto_optimization(analysis_results)
        
        # Generate final summary report
        summary_report = self._generate_summary_report(analysis_results, optimization_results)
        
        total_time = time.time() - start_time
        
        print(f"\n✅ COMPREHENSIVE ANALYSIS COMPLETE")
        print(f"⏱️  Total time: {total_time:.1f} seconds")
        print(f"📁 Results saved to: {self.output_dir}")
        
        return {
            'validation_results': all_results,
            'analysis': analysis_results,
            'optimization': optimization_results,
            'summary': summary_report,
            'execution_time': total_time
        }
    
    def _run_parallel_validation(self, configurations: List[str], fast_mode: bool) -> Dict[str, Any]:
        """Run validation for all configurations in parallel"""
        all_results = {}
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all configuration validations
            future_to_config = {
                executor.submit(self._validate_single_config, config, fast_mode): config
                for config in configurations
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    result = future.result()
                    all_results[config] = result
                    print(f"✅ {config:20} - MAE: ${result.overall_metrics['mae']:.2f}, R²: {result.overall_metrics['r2']:.3f}")
                except Exception as e:
                    print(f"❌ {config:20} - Failed: {e}")
                    
        return all_results
    
    def _validate_single_config(self, config_name: str, fast_mode: bool):
        """Validate a single configuration"""
        if fast_mode:
            return self.validator.validate_baseline_only(config_name)
        else:
            return self.validator.validate_configuration(config_name)
    
    def _generate_comprehensive_analysis(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive performance analysis"""
        
        # Create performance comparison dataframe
        performance_data = []
        
        for config_name, result in all_results.items():
            metrics = result.overall_metrics
            timing = result.timing_info
            geographic = result.geographic_analysis
            
            performance_data.append({
                'configuration': config_name,
                'mae': metrics['mae'],
                'rmse': metrics['rmse'],
                'r2': metrics['r2'],
                'mape': metrics.get('mape', 0),
                'validation_time': timing.get('total_time', 0),
                'accuracy_level': self._classify_accuracy(metrics['mae']),
                'speed_level': self._classify_speed(timing.get('total_time', 0)),
                'geographic_coverage': geographic.get('coverage', 0),
                'avg_confidence': 0.75,  # Default confidence for demo
                'total_predictions': sum(len(fold.get('predictions', [])) if isinstance(fold, dict) else 0 
                                       for fold in result.fold_results)
            })
        
        performance_df = pd.DataFrame(performance_data)
        
        # Calculate rankings
        performance_df['accuracy_rank'] = performance_df['mae'].rank()
        performance_df['speed_rank'] = performance_df['validation_time'].rank()
        performance_df['overall_rank'] = (performance_df['accuracy_rank'] + 
                                        performance_df['speed_rank']).rank()
        
        # Identify best configurations
        best_accuracy = performance_df.loc[performance_df['mae'].idxmin()]
        best_speed = performance_df.loc[performance_df['validation_time'].idxmin()]
        best_overall = performance_df.loc[performance_df['overall_rank'].idxmin()]
        
        return {
            'performance_df': performance_df,
            'best_accuracy': best_accuracy,
            'best_speed': best_speed,
            'best_overall': best_overall,
            'summary_stats': {
                'avg_mae': performance_df['mae'].mean(),
                'std_mae': performance_df['mae'].std(),
                'avg_time': performance_df['validation_time'].mean(),
                'std_time': performance_df['validation_time'].std(),
                'total_configs': len(performance_df)
            }
        }
    
    def _create_performance_dashboard(self, analysis_results: Dict[str, Any]):
        """Create comprehensive performance visualization dashboard"""
        performance_df = analysis_results['performance_df']
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create comprehensive dashboard
        fig = plt.figure(figsize=(20, 16))
        
        # 1. MAE Comparison
        plt.subplot(3, 3, 1)
        bars = plt.bar(range(len(performance_df)), performance_df['mae'], 
                      color=sns.color_palette("viridis", len(performance_df)))
        plt.title('Mean Absolute Error by Configuration', fontsize=14, fontweight='bold')
        plt.xlabel('Configuration')
        plt.ylabel('MAE ($)')
        plt.xticks(range(len(performance_df)), performance_df['configuration'], rotation=45, ha='right')
        
        # Add value labels on bars
        for i, (bar, mae) in enumerate(zip(bars, performance_df['mae'])):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'${mae:.2f}', ha='center', va='bottom', fontsize=10)
        
        # 2. R² Comparison
        plt.subplot(3, 3, 2)
        bars = plt.bar(range(len(performance_df)), performance_df['r2'],
                      color=sns.color_palette("plasma", len(performance_df)))
        plt.title('R² Score by Configuration', fontsize=14, fontweight='bold')
        plt.xlabel('Configuration')
        plt.ylabel('R² Score')
        plt.xticks(range(len(performance_df)), performance_df['configuration'], rotation=45, ha='right')
        
        # Add value labels
        for i, (bar, r2) in enumerate(zip(bars, performance_df['r2'])):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{r2:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 3. Speed vs Accuracy Scatter
        plt.subplot(3, 3, 3)
        scatter = plt.scatter(performance_df['validation_time'], performance_df['mae'], 
                             s=100, c=performance_df['r2'], cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='R² Score')
        plt.xlabel('Validation Time (seconds)')
        plt.ylabel('MAE ($)')
        plt.title('Speed vs Accuracy Trade-off', fontsize=14, fontweight='bold')
        
        # Add configuration labels
        for i, config in enumerate(performance_df['configuration']):
            plt.annotate(config[:8], (performance_df.iloc[i]['validation_time'], 
                        performance_df.iloc[i]['mae']), fontsize=8)
        
        # 4. Performance Radar Chart
        plt.subplot(3, 3, 4, projection='polar')
        
        # Normalize metrics for radar chart
        metrics_norm = performance_df[['mae', 'rmse', 'validation_time']].copy()
        metrics_norm['mae'] = 1 - (metrics_norm['mae'] / metrics_norm['mae'].max())  # Invert MAE (higher is better)
        metrics_norm['rmse'] = 1 - (metrics_norm['rmse'] / metrics_norm['rmse'].max())  # Invert RMSE
        metrics_norm['validation_time'] = 1 - (metrics_norm['validation_time'] / metrics_norm['validation_time'].max())  # Invert time
        
        # Plot top 3 configurations
        top_configs = performance_df.nsmallest(3, 'overall_rank')
        angles = np.linspace(0, 2 * np.pi, 3, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for idx, (_, config_row) in enumerate(top_configs.iterrows()):
            values = [
                1 - (config_row['mae'] / performance_df['mae'].max()),
                1 - (config_row['rmse'] / performance_df['rmse'].max()),
                1 - (config_row['validation_time'] / performance_df['validation_time'].max())
            ]
            values += values[:1]
            
            plt.plot(angles, values, 'o-', linewidth=2, label=config_row['configuration'][:10])
            plt.fill(angles, values, alpha=0.25)
        
        plt.xticks(angles[:-1], ['Accuracy\n(MAE)', 'Precision\n(RMSE)', 'Speed\n(Time)'])
        plt.ylim(0, 1)
        plt.title('Top 3 Configurations\nPerformance Radar', fontsize=14, fontweight='bold')
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # 5. Confidence Distribution
        plt.subplot(3, 3, 5)
        plt.hist(performance_df['avg_confidence'], bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Confidence Level Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Average Confidence (%)')
        plt.ylabel('Frequency')
        
        # 6. Geographic Coverage
        plt.subplot(3, 3, 6)
        bars = plt.bar(range(len(performance_df)), performance_df['geographic_coverage'],
                      color=sns.color_palette("coolwarm", len(performance_df)))
        plt.title('Geographic Coverage by Configuration', fontsize=14, fontweight='bold')
        plt.xlabel('Configuration')
        plt.ylabel('Coverage (locations)')
        plt.xticks(range(len(performance_df)), performance_df['configuration'], rotation=45, ha='right')
        
        # 7. Overall Rankings
        plt.subplot(3, 3, 7)
        ranking_data = performance_df[['configuration', 'accuracy_rank', 'speed_rank', 'overall_rank']].set_index('configuration')
        ranking_data.plot(kind='bar', ax=plt.gca())
        plt.title('Configuration Rankings', fontsize=14, fontweight='bold')
        plt.xlabel('Configuration')
        plt.ylabel('Rank (lower is better)')
        plt.xticks(rotation=45, ha='right')
        plt.legend(['Accuracy Rank', 'Speed Rank', 'Overall Rank'])
        
        # 8. Performance Heatmap
        plt.subplot(3, 3, 8)
        heatmap_data = performance_df[['configuration', 'mae', 'rmse', 'r2', 'validation_time']].set_index('configuration')
        # Normalize for heatmap
        heatmap_data_norm = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())
        sns.heatmap(heatmap_data_norm.T, annot=True, fmt='.2f', cmap='RdYlBu_r', ax=plt.gca())
        plt.title('Normalized Performance Heatmap', fontsize=14, fontweight='bold')
        plt.ylabel('Metrics')
        
        # 9. Summary Statistics
        plt.subplot(3, 3, 9)
        plt.axis('off')
        summary_stats = analysis_results['summary_stats']
        best_overall = analysis_results['best_overall']
        
        summary_text = f"""
PERFORMANCE SUMMARY
═══════════════════

📊 CONFIGURATIONS TESTED: {summary_stats['total_configs']}

🏆 BEST OVERALL: {best_overall['configuration']}
   • MAE: ${best_overall['mae']:.2f}
   • R²: {best_overall['r2']:.3f}
   • Time: {best_overall['validation_time']:.1f}s

📈 AVERAGE PERFORMANCE:
   • MAE: ${summary_stats['avg_mae']:.2f} ± ${summary_stats['std_mae']:.2f}
   • Time: {summary_stats['avg_time']:.1f}s ± {summary_stats['std_time']:.1f}s

🎯 ACCURACY LEADER: {analysis_results['best_accuracy']['configuration']}
   • MAE: ${analysis_results['best_accuracy']['mae']:.2f}

⚡ SPEED LEADER: {analysis_results['best_speed']['configuration']}
   • Time: {analysis_results['best_speed']['validation_time']:.1f}s
        """
        
        plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        
        # Save the dashboard
        dashboard_path = self.output_dir / "charts" / f"performance_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        print(f"📊 Performance dashboard saved: {dashboard_path}")
        
        plt.show()
        
        return dashboard_path
    
    def _generate_ai_readable_output(self, analysis_results: Dict[str, Any]):
        """Generate structured output for AI analysis"""
        
        # Convert DataFrame to JSON-serializable format
        performance_data = analysis_results['performance_df'].to_dict('records')
        
        ai_output = {
            'timestamp': datetime.now().isoformat(),
            'analysis_metadata': {
                'total_configurations': len(performance_data),
                'analysis_type': 'comprehensive_validation',
                'system_version': 'FastKFoldValidator_v2.0'
            },
            'performance_matrix': performance_data,
            'best_configurations': {
                'accuracy_leader': {
                    'name': analysis_results['best_accuracy']['configuration'],
                    'mae': float(analysis_results['best_accuracy']['mae']),
                    'r2': float(analysis_results['best_accuracy']['r2']),
                    'validation_time': float(analysis_results['best_accuracy']['validation_time'])
                },
                'speed_leader': {
                    'name': analysis_results['best_speed']['configuration'],
                    'mae': float(analysis_results['best_speed']['mae']),
                    'r2': float(analysis_results['best_speed']['r2']),
                    'validation_time': float(analysis_results['best_speed']['validation_time'])
                },
                'overall_leader': {
                    'name': analysis_results['best_overall']['configuration'],
                    'mae': float(analysis_results['best_overall']['mae']),
                    'r2': float(analysis_results['best_overall']['r2']),
                    'validation_time': float(analysis_results['best_overall']['validation_time'])
                }
            },
            'summary_statistics': {
                'mae_statistics': {
                    'mean': float(analysis_results['summary_stats']['avg_mae']),
                    'std': float(analysis_results['summary_stats']['std_mae']),
                    'min': float(analysis_results['performance_df']['mae'].min()),
                    'max': float(analysis_results['performance_df']['mae'].max())
                },
                'time_statistics': {
                    'mean': float(analysis_results['summary_stats']['avg_time']),
                    'std': float(analysis_results['summary_stats']['std_time']),
                    'min': float(analysis_results['performance_df']['validation_time'].min()),
                    'max': float(analysis_results['performance_df']['validation_time'].max())
                }
            },
            'optimization_targets': {
                'accuracy_threshold': 1.0,  # Target MAE < $1.00
                'speed_threshold': 2.0,     # Target validation < 2.0 seconds
                'r2_threshold': 0.8         # Target R² > 0.8
            }
        }
        
        # Save AI-readable output
        ai_output_path = self.output_dir / "data" / f"ai_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(ai_output_path, 'w') as f:
            json.dump(ai_output, f, indent=2)
        
        print(f"🤖 AI-readable analysis saved: {ai_output_path}")
        
        return ai_output_path
    
    def _run_auto_optimization(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run auto-optimization analysis and generate recommendations"""
        
        performance_df = analysis_results['performance_df']
        
        # Define optimization targets
        targets = {
            'mae_target': 1.0,       # Target MAE < $1.00
            'speed_target': 2.0,     # Target time < 2.0 seconds  
            'r2_target': 0.8         # Target R² > 0.8
        }
        
        # Analyze configurations meeting targets
        accuracy_achievers = performance_df[performance_df['mae'] < targets['mae_target']]
        speed_achievers = performance_df[performance_df['validation_time'] < targets['speed_target']]
        quality_achievers = performance_df[performance_df['r2'] > targets['r2_target']]
        
        # Find configurations meeting multiple targets
        all_targets = performance_df[
            (performance_df['mae'] < targets['mae_target']) &
            (performance_df['validation_time'] < targets['speed_target']) &
            (performance_df['r2'] > targets['r2_target'])
        ]
        
        # Generate optimization recommendations
        recommendations = []
        
        if len(all_targets) > 0:
            best_config = all_targets.loc[all_targets['overall_rank'].idxmin()]
            recommendations.append({
                'type': 'optimal_configuration',
                'priority': 'HIGH',
                'recommendation': f"Use '{best_config['configuration']}' as primary configuration",
                'reasoning': f"Meets all targets: MAE ${best_config['mae']:.2f}, R² {best_config['r2']:.3f}, Time {best_config['validation_time']:.1f}s",
                'implementation': f"Set default_config = '{best_config['configuration']}'"
            })
        else:
            # Recommend best trade-offs
            if len(accuracy_achievers) > 0:
                best_accuracy = accuracy_achievers.loc[accuracy_achievers['mae'].idxmin()]
                recommendations.append({
                    'type': 'accuracy_optimization',
                    'priority': 'MEDIUM',
                    'recommendation': f"For accuracy-critical tasks, use '{best_accuracy['configuration']}'",
                    'reasoning': f"Best accuracy: MAE ${best_accuracy['mae']:.2f}",
                    'implementation': f"accuracy_config = '{best_accuracy['configuration']}'"
                })
            
            if len(speed_achievers) > 0:
                fastest = speed_achievers.loc[speed_achievers['validation_time'].idxmin()]
                recommendations.append({
                    'type': 'speed_optimization',
                    'priority': 'MEDIUM',
                    'recommendation': f"For speed-critical tasks, use '{fastest['configuration']}'",
                    'reasoning': f"Fastest validation: {fastest['validation_time']:.1f}s",
                    'implementation': f"speed_config = '{fastest['configuration']}'"
                })
        
        # Identify underperforming configurations
        worst_performers = performance_df.nlargest(2, 'overall_rank')
        for _, config in worst_performers.iterrows():
            recommendations.append({
                'type': 'deprecation_candidate',
                'priority': 'LOW',
                'recommendation': f"Consider deprecating '{config['configuration']}'",
                'reasoning': f"Poor performance: MAE ${config['mae']:.2f}, Time {config['validation_time']:.1f}s",
                'implementation': f"# Remove or improve '{config['configuration']}' configuration"
            })
        
        # Generate auto-tuning suggestions
        mae_mean = performance_df['mae'].mean()
        mae_std = performance_df['mae'].std()
        
        if mae_mean > 1.5:
            recommendations.append({
                'type': 'parameter_tuning',
                'priority': 'HIGH',
                'recommendation': "Overall accuracy needs improvement",
                'reasoning': f"Average MAE ${mae_mean:.2f} exceeds acceptable threshold",
                'implementation': "Consider: Increase k_neighbors, expand historical_days, add more features"
            })
        
        optimization_results = {
            'targets': targets,
            'achievement_summary': {
                'accuracy_achievers': len(accuracy_achievers),
                'speed_achievers': len(speed_achievers),
                'quality_achievers': len(quality_achievers),
                'all_targets_achievers': len(all_targets)
            },
            'recommendations': recommendations,
            'performance_gaps': {
                'accuracy_gap': max(0, performance_df['mae'].min() - targets['mae_target']),
                'speed_gap': max(0, performance_df['validation_time'].min() - targets['speed_target']),
                'quality_gap': max(0, targets['r2_target'] - performance_df['r2'].max())
            }
        }
        
        return optimization_results
    
    def _generate_summary_report(self, analysis_results: Dict[str, Any], 
                                optimization_results: Dict[str, Any]) -> str:
        """Generate comprehensive summary report"""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        performance_df = analysis_results['performance_df']
        best_overall = analysis_results['best_overall']
        
        report = f"""
# 🚀 COMPREHENSIVE VALIDATION ANALYSIS REPORT
Generated: {timestamp}

## 📊 EXECUTIVE SUMMARY
═══════════════════════════════════════════════════════════════════════

🎯 **CONFIGURATIONS ANALYZED:** {len(performance_df)}
⚡ **BEST OVERALL CONFIGURATION:** {best_overall['configuration']}
   • Accuracy (MAE): ${best_overall['mae']:.2f}
   • Correlation (R²): {best_overall['r2']:.3f}  
   • Validation Speed: {best_overall['validation_time']:.1f} seconds

## 🏆 PERFORMANCE LEADERS
═══════════════════════════════════════════════════════════════════════

🎯 **ACCURACY CHAMPION:** {analysis_results['best_accuracy']['configuration']}
   • MAE: ${analysis_results['best_accuracy']['mae']:.2f}
   • R²: {analysis_results['best_accuracy']['r2']:.3f}

⚡ **SPEED CHAMPION:** {analysis_results['best_speed']['configuration']}  
   • Validation Time: {analysis_results['best_speed']['validation_time']:.1f}s
   • MAE: ${analysis_results['best_speed']['mae']:.2f}

## 📈 PERFORMANCE STATISTICS
═══════════════════════════════════════════════════════════════════════

**Accuracy Metrics:**
   • Average MAE: ${analysis_results['summary_stats']['avg_mae']:.2f} ± ${analysis_results['summary_stats']['std_mae']:.2f}
   • Best MAE: ${performance_df['mae'].min():.2f}
   • Worst MAE: ${performance_df['mae'].max():.2f}

**Speed Metrics:**
   • Average Time: {analysis_results['summary_stats']['avg_time']:.1f}s ± {analysis_results['summary_stats']['std_time']:.1f}s
   • Fastest: {performance_df['validation_time'].min():.1f}s
   • Slowest: {performance_df['validation_time'].max():.1f}s

## 🔧 AUTO-OPTIMIZATION ANALYSIS
═══════════════════════════════════════════════════════════════════════

**Target Achievement:**
   • Accuracy Target (MAE < $1.00): {optimization_results['achievement_summary']['accuracy_achievers']}/{len(performance_df)} configurations
   • Speed Target (< 2.0s): {optimization_results['achievement_summary']['speed_achievers']}/{len(performance_df)} configurations  
   • Quality Target (R² > 0.8): {optimization_results['achievement_summary']['quality_achievers']}/{len(performance_df)} configurations
   • All Targets Met: {optimization_results['achievement_summary']['all_targets_achievers']}/{len(performance_df)} configurations

## 🚀 OPTIMIZATION RECOMMENDATIONS
═══════════════════════════════════════════════════════════════════════
"""

        # Add recommendations
        for i, rec in enumerate(optimization_results['recommendations'], 1):
            priority_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
            report += f"""
**{i}. {rec['type'].replace('_', ' ').title()}** {priority_emoji.get(rec['priority'], '⚪')}
   • **Recommendation:** {rec['recommendation']}
   • **Reasoning:** {rec['reasoning']}
   • **Implementation:** `{rec['implementation']}`
"""

        report += f"""

## 📋 DETAILED CONFIGURATION PERFORMANCE
═══════════════════════════════════════════════════════════════════════

"""
        
        # Add detailed performance table
        for _, config in performance_df.iterrows():
            report += f"""
**{config['configuration']}**
   • MAE: ${config['mae']:.2f} | RMSE: ${config['rmse']:.2f} | R²: {config['r2']:.3f}
   • Speed: {config['validation_time']:.1f}s | Confidence: {config['avg_confidence']:.1f}%
   • Rank: #{int(config['overall_rank'])} overall
"""

        report += f"""

## 💡 SYSTEM INSIGHTS
═══════════════════════════════════════════════════════════════════════

🔍 **Performance Patterns:**
   • Configuration diversity shows {performance_df['mae'].std():.2f} standard deviation in MAE
   • Speed variation: {performance_df['validation_time'].std():.1f}s standard deviation
   • Best accuracy/speed ratio: {(performance_df['mae'].min() / performance_df['validation_time'].min()):.3f}

🎯 **Business Impact:**
   • Estimated cost savings from optimization: ${((analysis_results['summary_stats']['avg_mae'] - performance_df['mae'].min()) * 1000):.0f} per 1000 predictions
   • Time savings potential: {((analysis_results['summary_stats']['avg_time'] - performance_df['validation_time'].min()) * 100):.0f}s per 100 validations

📁 **Generated Files:**
   • Performance Dashboard: /charts/performance_dashboard_*.png
   • AI Analysis Data: /data/ai_analysis_*.json  
   • Summary Report: /reports/summary_report_*.md

═══════════════════════════════════════════════════════════════════════
Report generated by ComprehensiveValidationOrchestrator v2.0
"""
        
        # Save the report
        report_path = self.output_dir / "reports" / f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📋 Summary report saved: {report_path}")
        print("\n" + "="*60)
        print(report)
        
        return report
    
    def _classify_accuracy(self, mae: float) -> str:
        """Classify accuracy level based on MAE"""
        if mae < 0.5:
            return "EXCELLENT"
        elif mae < 1.0:
            return "GOOD"
        elif mae < 1.5:
            return "FAIR"
        else:
            return "NEEDS_IMPROVEMENT"
    
    def _classify_speed(self, time: float) -> str:
        """Classify speed level based on validation time"""
        if time < 1.0:
            return "VERY_FAST"
        elif time < 2.0:
            return "FAST"  
        elif time < 5.0:
            return "MODERATE"
        else:
            return "SLOW"

# Auto-optimization configuration generator
def generate_optimized_config(analysis_results: Dict[str, Any]) -> str:
    """Generate optimized configuration code based on analysis results"""
    
    best_config = analysis_results['best_overall']['configuration']
    
    config_code = f"""
# AUTO-GENERATED OPTIMIZED CONFIGURATION
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Based on comprehensive validation analysis

class OptimizedConfig:
    '''
    Auto-optimized configuration based on comprehensive validation analysis
    Performance: MAE ${analysis_results['best_overall']['mae']:.2f}, R² {analysis_results['best_overall']['r2']:.3f}
    '''
    
    # Primary configuration (best overall performance)
    PRIMARY_CONFIG = '{best_config}'
    
    # Specialized configurations for different use cases
    ACCURACY_OPTIMIZED = '{analysis_results['best_accuracy']['configuration']}'
    SPEED_OPTIMIZED = '{analysis_results['best_speed']['configuration']}'
    
    # Performance thresholds for auto-switching
    MAE_THRESHOLD = {analysis_results['best_overall']['mae']:.2f}
    SPEED_THRESHOLD = {analysis_results['best_overall']['validation_time']:.1f}
    R2_THRESHOLD = {analysis_results['best_overall']['r2']:.3f}
    
    @staticmethod
    def get_config_for_use_case(use_case: str) -> str:
        '''Get optimal configuration for specific use case'''
        if use_case == 'accuracy_critical':
            return OptimizedConfig.ACCURACY_OPTIMIZED
        elif use_case == 'speed_critical':  
            return OptimizedConfig.SPEED_OPTIMIZED
        else:
            return OptimizedConfig.PRIMARY_CONFIG
"""
    
    return config_code

if __name__ == "__main__":
    # Demo the comprehensive validation system
    orchestrator = ComprehensiveValidationOrchestrator()
    
    print("🚀 Starting Comprehensive Validation Analysis...")
    results = orchestrator.run_comprehensive_analysis(fast_mode=True)
    
    print("\n✅ Analysis complete! Check the output directory for detailed results.")
