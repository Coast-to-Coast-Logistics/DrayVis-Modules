"""
Auto-Optimization Engine
=======================

Intelligent system that analyzes validation results and automatically
optimizes the system for better performance. Features:

1. Performance pattern analysis
2. Automatic parameter tuning
3. Configuration recommendation
4. Self-improving algorithms
5. Adaptive learning from results
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
from pathlib import Path
import logging

class AutoOptimizationEngine:
    """
    Intelligent optimization engine that learns from validation results
    and automatically improves system performance
    """
    
    def __init__(self, learning_rate: float = 0.1, memory_window: int = 10):
        """
        Initialize the auto-optimization engine
        
        Args:
            learning_rate: How quickly to adapt to new results (0.0 to 1.0)
            memory_window: Number of previous results to consider for learning
        """
        self.learning_rate = learning_rate
        self.memory_window = memory_window
        self.performance_history = []
        self.optimization_log = []
        
        # Performance targets (can be dynamically adjusted)
        self.targets = {
            'mae_target': 1.0,      # Target MAE < $1.00
            'speed_target': 2.0,    # Target validation < 2.0 seconds
            'r2_target': 0.8,       # Target R² > 0.8
            'confidence_target': 80.0  # Target confidence > 80%
        }
        
        # Learning parameters
        self.parameter_bounds = {
            'k_neighbors': (3, 20),
            'historical_days': (30, 365),
            'confidence_threshold': (0.5, 0.95),
            'distance_weight': (0.1, 2.0),
            'time_decay_factor': (0.1, 1.0)
        }
        
        print("🧠 Auto-Optimization Engine Initialized")
        print(f"📊 Learning Rate: {learning_rate}")
        print(f"🔄 Memory Window: {memory_window} results")
    
    def analyze_and_optimize(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze validation results and generate optimization recommendations
        
        Args:
            analysis_results: Results from comprehensive validation analysis
            
        Returns:
            Dictionary containing optimization analysis and recommendations
        """
        print("\n🧠 AUTO-OPTIMIZATION ANALYSIS")
        print("=" * 50)
        
        # Store current results in memory
        self._update_performance_history(analysis_results)
        
        # Analyze performance patterns
        pattern_analysis = self._analyze_performance_patterns()
        
        # Generate parameter optimization suggestions
        parameter_optimizations = self._optimize_parameters(analysis_results)
        
        # Identify configuration improvements
        config_improvements = self._identify_configuration_improvements(analysis_results)
        
        # Generate adaptive recommendations
        adaptive_recommendations = self._generate_adaptive_recommendations(
            pattern_analysis, parameter_optimizations, config_improvements
        )
        
        # Create self-tuning parameters
        auto_tuned_params = self._generate_auto_tuned_parameters(analysis_results)
        
        optimization_result = {
            'timestamp': datetime.now().isoformat(),
            'pattern_analysis': pattern_analysis,
            'parameter_optimizations': parameter_optimizations,
            'configuration_improvements': config_improvements,
            'adaptive_recommendations': adaptive_recommendations,
            'auto_tuned_parameters': auto_tuned_params,
            'learning_insights': self._generate_learning_insights()
        }
        
        # Log this optimization cycle
        self.optimization_log.append(optimization_result)
        
        print("✅ Auto-optimization analysis complete")
        return optimization_result
    
    def _update_performance_history(self, analysis_results: Dict[str, Any]):
        """Update performance history with latest results"""
        
        # Check if performance_df exists in analysis_results or nested in analysis
        performance_df = None
        if 'performance_df' in analysis_results:
            performance_df = analysis_results['performance_df']
        elif 'analysis' in analysis_results and 'performance_df' in analysis_results['analysis']:
            performance_df = analysis_results['analysis']['performance_df']
        else:
            print("❌ Warning: performance_df not found in analysis_results")
            print(f"   Available keys: {list(analysis_results.keys())}")
            if 'analysis' in analysis_results:
                print(f"   Analysis keys: {list(analysis_results['analysis'].keys())}")
            return
        
        current_performance = {
            'timestamp': datetime.now().isoformat(),
            'best_mae': float(performance_df['mae'].min()),
            'avg_mae': float(performance_df['mae'].mean()),
            'best_r2': float(performance_df['r2'].max()),
            'avg_r2': float(performance_df['r2'].mean()),
            'best_speed': float(performance_df['validation_time'].min()),
            'avg_speed': float(performance_df['validation_time'].mean()),
            'config_count': len(performance_df)
        }
        
        # Get best_overall safely
        best_overall = None
        if 'best_overall' in analysis_results:
            best_overall = analysis_results['best_overall']
        elif 'analysis' in analysis_results and 'best_overall' in analysis_results['analysis']:
            best_overall = analysis_results['analysis']['best_overall']
            
        if best_overall is not None:
            if isinstance(best_overall, dict) and 'configuration' in best_overall:
                current_performance['best_overall_config'] = best_overall['configuration']
            elif hasattr(best_overall, 'configuration'):
                current_performance['best_overall_config'] = best_overall.configuration
            else:
                current_performance['best_overall_config'] = str(best_overall)
        else:
            current_performance['best_overall_config'] = 'unknown'
        
        self.performance_history.append(current_performance)
        
        # Keep only recent history within memory window
        if len(self.performance_history) > self.memory_window:
            self.performance_history = self.performance_history[-self.memory_window:]
    
    def _analyze_performance_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in performance history"""
        
        if len(self.performance_history) < 2:
            return {
                'trend_analysis': 'Insufficient data for trend analysis',
                'performance_stability': 'Unknown',
                'improvement_rate': 0.0
            }
        
        # Convert history to DataFrame for analysis
        history_df = pd.DataFrame(self.performance_history)
        
        # Analyze trends
        mae_trend = np.polyfit(range(len(history_df)), history_df['best_mae'], 1)[0]
        r2_trend = np.polyfit(range(len(history_df)), history_df['best_r2'], 1)[0]
        speed_trend = np.polyfit(range(len(history_df)), history_df['best_speed'], 1)[0]
        
        # Calculate stability (coefficient of variation)
        mae_stability = history_df['best_mae'].std() / history_df['best_mae'].mean()
        r2_stability = history_df['best_r2'].std() / history_df['best_r2'].mean()
        
        # Calculate improvement rate
        if len(history_df) >= 3:
            recent_mae = history_df['best_mae'].iloc[-3:].mean()
            older_mae = history_df['best_mae'].iloc[:-3].mean() if len(history_df) > 3 else recent_mae
            improvement_rate = (older_mae - recent_mae) / older_mae if older_mae > 0 else 0.0
        else:
            improvement_rate = 0.0
        
        return {
            'trend_analysis': {
                'mae_trend': float(mae_trend),  # Negative is good (decreasing error)
                'r2_trend': float(r2_trend),   # Positive is good (increasing correlation)
                'speed_trend': float(speed_trend),  # Negative is good (getting faster)
                'interpretation': self._interpret_trends(mae_trend, r2_trend, speed_trend)
            },
            'performance_stability': {
                'mae_stability': float(mae_stability),
                'r2_stability': float(r2_stability),
                'overall_stability': 'High' if max(mae_stability, r2_stability) < 0.1 else 'Medium' if max(mae_stability, r2_stability) < 0.2 else 'Low'
            },
            'improvement_rate': float(improvement_rate),
            'data_points': len(history_df)
        }
    
    def _interpret_trends(self, mae_trend: float, r2_trend: float, speed_trend: float) -> str:
        """Interpret performance trends"""
        
        improvements = []
        concerns = []
        
        if mae_trend < -0.01:  # MAE decreasing significantly
            improvements.append("accuracy improving")
        elif mae_trend > 0.01:  # MAE increasing
            concerns.append("accuracy declining")
        
        if r2_trend > 0.01:  # R² increasing
            improvements.append("correlation improving")
        elif r2_trend < -0.01:  # R² decreasing
            concerns.append("correlation declining")
            
        if speed_trend < -0.1:  # Speed improving (time decreasing)
            improvements.append("speed improving")
        elif speed_trend > 0.1:  # Speed declining (time increasing)
            concerns.append("speed declining")
        
        if improvements and not concerns:
            return f"Positive trends: {', '.join(improvements)}"
        elif concerns and not improvements:
            return f"Concerning trends: {', '.join(concerns)}"
        elif improvements and concerns:
            return f"Mixed trends - Improvements: {', '.join(improvements)}; Concerns: {', '.join(concerns)}"
        else:
            return "Performance stable with minimal changes"
    
    def _optimize_parameters(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate parameter optimization suggestions based on performance analysis"""
        
        # Get performance_df from analysis_results
        performance_df = None
        if 'performance_df' in analysis_results:
            performance_df = analysis_results['performance_df']
        elif 'analysis' in analysis_results and 'performance_df' in analysis_results['analysis']:
            performance_df = analysis_results['analysis']['performance_df']
        else:
            print("❌ Warning: performance_df not found for parameter optimization")
            return {}
            
        # Get best_overall from analysis_results
        best_config = None
        if 'best_overall' in analysis_results:
            best_config = analysis_results['best_overall']
        elif 'analysis' in analysis_results and 'best_overall' in analysis_results['analysis']:
            best_config = analysis_results['analysis']['best_overall']
        else:
            print("❌ Warning: best_overall not found for parameter optimization")
            return {}
        
        optimizations = {}
        
        # Analyze accuracy performance
        if best_config['mae'] > self.targets['mae_target']:
            accuracy_gap = best_config['mae'] - self.targets['mae_target']
            
            optimizations['accuracy_improvements'] = {
                'current_mae': float(best_config['mae']),
                'target_mae': self.targets['mae_target'],
                'gap': float(accuracy_gap),
                'suggested_adjustments': []
            }
            
            # Suggest parameter adjustments based on gap size
            if accuracy_gap > 0.5:  # Large gap
                optimizations['accuracy_improvements']['suggested_adjustments'].extend([
                    {'parameter': 'k_neighbors', 'adjustment': 'increase', 'reason': 'More neighbors for better averaging'},
                    {'parameter': 'historical_days', 'adjustment': 'increase', 'reason': 'More historical data for patterns'},
                    {'parameter': 'feature_engineering', 'adjustment': 'enhance', 'reason': 'Add seasonal and trend features'}
                ])
            elif accuracy_gap > 0.2:  # Medium gap
                optimizations['accuracy_improvements']['suggested_adjustments'].extend([
                    {'parameter': 'distance_weighting', 'adjustment': 'tune', 'reason': 'Optimize geographic weighting'},
                    {'parameter': 'confidence_threshold', 'adjustment': 'adjust', 'reason': 'Filter low-confidence predictions'}
                ])
            else:  # Small gap
                optimizations['accuracy_improvements']['suggested_adjustments'].append(
                    {'parameter': 'fine_tuning', 'adjustment': 'minor', 'reason': 'Small adjustments to existing parameters'}
                )
        
        # Analyze speed performance
        if best_config['validation_time'] > self.targets['speed_target']:
            speed_gap = best_config['validation_time'] - self.targets['speed_target']
            
            optimizations['speed_improvements'] = {
                'current_time': float(best_config['validation_time']),
                'target_time': self.targets['speed_target'],
                'gap': float(speed_gap),
                'suggested_adjustments': []
            }
            
            if speed_gap > 2.0:  # Significant speed issue
                optimizations['speed_improvements']['suggested_adjustments'].extend([
                    {'parameter': 'parallel_processing', 'adjustment': 'increase', 'reason': 'More CPU cores utilization'},
                    {'parameter': 'data_caching', 'adjustment': 'implement', 'reason': 'Cache frequently accessed data'},
                    {'parameter': 'algorithm_optimization', 'adjustment': 'review', 'reason': 'Consider faster algorithms'}
                ])
            else:  # Minor speed optimization
                optimizations['speed_improvements']['suggested_adjustments'].append(
                    {'parameter': 'batch_processing', 'adjustment': 'optimize', 'reason': 'Improve batch sizes'}
                )
        
        # Analyze R² performance
        if best_config['r2'] < self.targets['r2_target']:
            correlation_gap = self.targets['r2_target'] - best_config['r2']
            
            optimizations['correlation_improvements'] = {
                'current_r2': float(best_config['r2']),
                'target_r2': self.targets['r2_target'],
                'gap': float(correlation_gap),
                'suggested_adjustments': [
                    {'parameter': 'feature_selection', 'adjustment': 'enhance', 'reason': 'Better predictive features'},
                    {'parameter': 'model_complexity', 'adjustment': 'increase', 'reason': 'Capture more complex patterns'},
                    {'parameter': 'data_quality', 'adjustment': 'improve', 'reason': 'Clean and validate input data'}
                ]
            }
        
        return optimizations
    
    def _identify_configuration_improvements(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Identify specific configuration improvements"""
        
        # Get performance_df from analysis_results
        performance_df = None
        if 'performance_df' in analysis_results:
            performance_df = analysis_results['performance_df']
        elif 'analysis' in analysis_results and 'performance_df' in analysis_results['analysis']:
            performance_df = analysis_results['analysis']['performance_df']
        else:
            print("❌ Warning: performance_df not found for configuration improvements")
            return {
                'underperforming_configs': [],
                'hybrid_opportunities': [],
                'parameter_combinations': []
            }
        
        # Find configurations that could be merged or improved
        improvements = {
            'underperforming_configs': [],
            'hybrid_opportunities': [],
            'parameter_combinations': []
        }
        
        # Identify underperforming configurations
        mae_threshold = performance_df['mae'].quantile(0.75)  # Worst 25%
        underperformers = performance_df[performance_df['mae'] > mae_threshold]
        
        for _, config in underperformers.iterrows():
            improvements['underperforming_configs'].append({
                'name': config['configuration'],
                'mae': float(config['mae']),
                'suggested_action': 'review_parameters' if config['mae'] > performance_df['mae'].mean() + performance_df['mae'].std() else 'minor_tuning'
            })
        
        # Identify hybrid opportunities (combine strengths of different configs)
        # Extract from nested analysis structure
        if 'best_accuracy' in analysis_results:
            best_accuracy = analysis_results['best_accuracy']
            best_speed = analysis_results['best_speed']
        elif 'analysis' in analysis_results:
            best_accuracy = analysis_results['analysis']['best_accuracy']
            best_speed = analysis_results['analysis']['best_speed']
        else:
            print("❌ Warning: best_accuracy/best_speed not found for hybrid opportunities")
            best_accuracy = None
            best_speed = None
        
        if best_accuracy is not None and best_speed is not None and best_accuracy['configuration'] != best_speed['configuration']:
            improvements['hybrid_opportunities'].append({
                'type': 'accuracy_speed_hybrid',
                'accuracy_config': best_accuracy['configuration'],
                'speed_config': best_speed['configuration'],
                'suggestion': 'Create hybrid combining accuracy features with speed optimizations'
            })
        
        # Suggest parameter combinations based on top performers
        top_configs = performance_df.nsmallest(3, 'mae')
        improvements['parameter_combinations'] = [
            {
                'config': config['configuration'],
                'mae': float(config['mae']),
                'speed': float(config['validation_time']),
                'suggested_use': 'production' if config['mae'] < 1.0 and config['validation_time'] < 2.0 else 'specialized'
            }
            for _, config in top_configs.iterrows()
        ]
        
        return improvements
    
    def _generate_adaptive_recommendations(self, pattern_analysis: Dict[str, Any],
                                         parameter_optimizations: Dict[str, Any],
                                         config_improvements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate adaptive recommendations based on all analysis"""
        
        recommendations = []
        
        # Trend-based recommendations
        if 'trend_analysis' in pattern_analysis:
            trends = pattern_analysis['trend_analysis']
            
            # Check if trends is a dictionary (not a string when insufficient data)
            if isinstance(trends, dict):
                if trends['mae_trend'] > 0.01:  # Accuracy getting worse
                    recommendations.append({
                        'type': 'trend_correction',
                        'priority': 'HIGH',
                        'title': 'Accuracy Degradation Detected',
                        'description': 'Model accuracy is declining over time',
                        'action': 'Retrain models with recent data and review feature engineering',
                        'implementation': 'Schedule retraining every 30 days'
                    })
                
                if trends['speed_trend'] > 0.1:  # Getting slower
                    recommendations.append({
                        'type': 'performance_optimization',
                        'priority': 'MEDIUM',
                        'title': 'Speed Degradation Detected',
                        'description': 'Validation time is increasing',
                        'action': 'Profile code and optimize bottlenecks',
                        'implementation': 'Review data loading and processing efficiency'
                    })
            else:
                # Handle case where trend_analysis is a string (insufficient data)
                recommendations.append({
                    'type': 'data_collection',
                    'priority': 'LOW',
                    'title': 'Insufficient Historical Data',
                    'description': 'Need more validation runs to analyze performance trends',
                    'action': 'Continue running validations to build performance history',
                    'implementation': 'Run validation process multiple times over different time periods'
                })
        
        # Parameter-based recommendations
        if 'accuracy_improvements' in parameter_optimizations:
            acc_imp = parameter_optimizations['accuracy_improvements']
            if acc_imp['gap'] > 0.3:
                recommendations.append({
                    'type': 'parameter_tuning',
                    'priority': 'HIGH',
                    'title': 'Significant Accuracy Gap',
                    'description': f"Current MAE ${acc_imp['current_mae']:.2f} vs target ${acc_imp['target_mae']:.2f}",
                    'action': 'Implement suggested parameter adjustments',
                    'implementation': f"Apply {len(acc_imp['suggested_adjustments'])} parameter changes"
                })
        
        # Configuration-based recommendations
        if config_improvements['underperforming_configs']:
            recommendations.append({
                'type': 'configuration_cleanup',
                'priority': 'LOW',
                'title': 'Configuration Optimization',
                'description': f"Found {len(config_improvements['underperforming_configs'])} underperforming configurations",
                'action': 'Review and improve or deprecate poor-performing configurations',
                'implementation': 'Update configuration parameters or remove from active set'
            })
        
        # Stability-based recommendations
        if 'performance_stability' in pattern_analysis:
            stability = pattern_analysis['performance_stability']
            # Check if stability is a dictionary (not a string when insufficient data)
            if isinstance(stability, dict) and stability.get('overall_stability') == 'Low':
                recommendations.append({
                    'type': 'stability_improvement',
                    'priority': 'MEDIUM',
                    'title': 'Performance Instability',
                    'description': 'High variance in validation results',
                    'action': 'Implement more robust validation methodology',
                    'implementation': 'Increase fold count or add cross-validation seeds'
                })
        
        # Auto-tuning recommendations
        recommendations.append({
            'type': 'auto_tuning',
            'priority': 'MEDIUM',
            'title': 'Enable Continuous Optimization',
            'description': 'Implement automatic parameter tuning based on performance feedback',
            'action': 'Deploy auto-tuning system for continuous improvement',
            'implementation': 'Set up automated A/B testing of parameter variations'
        })
        
        return recommendations
    
    def _generate_auto_tuned_parameters(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate automatically tuned parameters based on current performance"""
        
        # Extract from nested analysis structure
        analysis_data = analysis_results.get('analysis', analysis_results)
        performance_df = analysis_data['performance_df']
        best_config = analysis_data['best_overall']
        
        # Base parameters (starting point)
        base_params = {
            'k_neighbors': 5,
            'historical_days': 90,
            'confidence_threshold': 0.7,
            'distance_weight': 1.0,
            'time_decay_factor': 0.5
        }
        
        # Adjust parameters based on performance
        tuned_params = base_params.copy()
        
        # Accuracy-based adjustments
        if best_config['mae'] > self.targets['mae_target']:
            mae_ratio = best_config['mae'] / self.targets['mae_target']
            
            # Increase neighbors for better stability
            tuned_params['k_neighbors'] = min(
                self.parameter_bounds['k_neighbors'][1],
                int(base_params['k_neighbors'] * (1 + (mae_ratio - 1) * 0.5))
            )
            
            # Increase historical data
            tuned_params['historical_days'] = min(
                self.parameter_bounds['historical_days'][1],
                int(base_params['historical_days'] * (1 + (mae_ratio - 1) * 0.3))
            )
        
        # Speed-based adjustments
        if best_config['validation_time'] > self.targets['speed_target']:
            speed_ratio = best_config['validation_time'] / self.targets['speed_target']
            
            # Reduce neighbors for speed
            tuned_params['k_neighbors'] = max(
                self.parameter_bounds['k_neighbors'][0],
                int(tuned_params['k_neighbors'] / (1 + (speed_ratio - 1) * 0.3))
            )
            
            # Reduce historical days for speed
            tuned_params['historical_days'] = max(
                self.parameter_bounds['historical_days'][0],
                int(tuned_params['historical_days'] / (1 + (speed_ratio - 1) * 0.2))
            )
        
        # R²-based adjustments
        if best_config['r2'] < self.targets['r2_target']:
            r2_gap = self.targets['r2_target'] - best_config['r2']
            
            # Adjust confidence threshold
            tuned_params['confidence_threshold'] = min(
                self.parameter_bounds['confidence_threshold'][1],
                base_params['confidence_threshold'] + r2_gap * 0.2
            )
            
            # Adjust distance weighting
            tuned_params['distance_weight'] = min(
                self.parameter_bounds['distance_weight'][1],
                base_params['distance_weight'] * (1 + r2_gap)
            )
        
        return {
            'base_parameters': base_params,
            'tuned_parameters': tuned_params,
            'adjustment_ratios': {
                'k_neighbors_change': tuned_params['k_neighbors'] / base_params['k_neighbors'],
                'historical_days_change': tuned_params['historical_days'] / base_params['historical_days'],
                'confidence_threshold_change': tuned_params['confidence_threshold'] / base_params['confidence_threshold']
            },
            'performance_drivers': {
                'mae_influence': best_config['mae'] / self.targets['mae_target'],
                'speed_influence': best_config['validation_time'] / self.targets['speed_target'],
                'r2_influence': self.targets['r2_target'] / max(best_config['r2'], 0.1)
            }
        }
    
    def _generate_learning_insights(self) -> Dict[str, Any]:
        """Generate insights from the learning process"""
        
        insights = {
            'optimization_cycles': len(self.optimization_log),
            'memory_utilization': len(self.performance_history),
            'learning_effectiveness': 'Unknown'
        }
        
        if len(self.performance_history) >= 3:
            # Calculate learning effectiveness
            recent_performance = np.mean([h['best_mae'] for h in self.performance_history[-2:]])
            older_performance = np.mean([h['best_mae'] for h in self.performance_history[:-2]])
            
            if recent_performance < older_performance:
                improvement = (older_performance - recent_performance) / older_performance
                if improvement > 0.1:
                    insights['learning_effectiveness'] = 'High'
                elif improvement > 0.05:
                    insights['learning_effectiveness'] = 'Medium'
                else:
                    insights['learning_effectiveness'] = 'Low'
            else:
                insights['learning_effectiveness'] = 'Declining'
            
            insights['improvement_rate'] = float(improvement) if 'improvement' in locals() else 0.0
        
        # Learning trajectory
        if len(self.performance_history) >= 2:
            mae_history = [h['best_mae'] for h in self.performance_history]
            insights['learning_trajectory'] = {
                'initial_mae': float(mae_history[0]),
                'current_mae': float(mae_history[-1]),
                'total_improvement': float((mae_history[0] - mae_history[-1]) / mae_history[0]) if mae_history[0] > 0 else 0.0,
                'consistency': float(np.std(mae_history) / np.mean(mae_history)) if np.mean(mae_history) > 0 else 0.0
            }
        
        return insights
    
    def export_optimized_configuration(self, optimization_results: Dict[str, Any]) -> str:
        """Export optimized configuration as executable code"""
        
        tuned_params = optimization_results['auto_tuned_parameters']['tuned_parameters']
        
        config_code = f"""
# AUTO-OPTIMIZED CONFIGURATION
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Learning Cycles: {len(self.optimization_log)}

class AutoOptimizedEstimatorConfig:
    '''
    Automatically optimized configuration based on performance learning
    This configuration adapts based on validation results and performance patterns
    '''
    
    # Core parameters (auto-tuned)
    K_NEIGHBORS = {tuned_params['k_neighbors']}
    HISTORICAL_DAYS = {tuned_params['historical_days']}
    CONFIDENCE_THRESHOLD = {tuned_params['confidence_threshold']:.3f}
    DISTANCE_WEIGHT = {tuned_params['distance_weight']:.3f}
    TIME_DECAY_FACTOR = {tuned_params['time_decay_factor']:.3f}
    
    # Performance targets
    TARGET_MAE = {self.targets['mae_target']:.2f}
    TARGET_SPEED = {self.targets['speed_target']:.1f}
    TARGET_R2 = {self.targets['r2_target']:.3f}
    
    # Learning parameters
    LEARNING_RATE = {self.learning_rate:.3f}
    MEMORY_WINDOW = {self.memory_window}
    
    @classmethod
    def get_adaptive_config(cls, current_performance: dict) -> dict:
        '''
        Get adaptive configuration based on current performance
        
        Args:
            current_performance: Dict with 'mae', 'r2', 'speed' keys
            
        Returns:
            Adjusted configuration parameters
        '''
        config = {{
            'k_neighbors': cls.K_NEIGHBORS,
            'historical_days': cls.HISTORICAL_DAYS,
            'confidence_threshold': cls.CONFIDENCE_THRESHOLD,
            'distance_weight': cls.DISTANCE_WEIGHT,
            'time_decay_factor': cls.TIME_DECAY_FACTOR
        }}
        
        # Adaptive adjustments based on current performance
        if current_performance.get('mae', 0) > cls.TARGET_MAE:
            config['k_neighbors'] = min(20, int(config['k_neighbors'] * 1.2))
            config['historical_days'] = min(365, int(config['historical_days'] * 1.1))
        
        if current_performance.get('speed', 0) > cls.TARGET_SPEED:
            config['k_neighbors'] = max(3, int(config['k_neighbors'] * 0.9))
            config['historical_days'] = max(30, int(config['historical_days'] * 0.95))
            
        if current_performance.get('r2', 0) < cls.TARGET_R2:
            config['confidence_threshold'] = min(0.95, config['confidence_threshold'] + 0.05)
            
        return config
"""
        
        return config_code

if __name__ == "__main__":
    # Demo the auto-optimization engine
    engine = AutoOptimizationEngine()
    print("🧠 Auto-Optimization Engine initialized and ready for analysis")
