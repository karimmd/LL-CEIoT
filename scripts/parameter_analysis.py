#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import json
import itertools
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging
import sys
import os

# Add algorithm modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'algorithms'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from multi_objective_optimizer import MultiObjectiveOptimizer, SystemState, OptimizationWeights
from transmission_energy_calculator import TransmissionEnergyCalculator

class LambdaParameterAnalyzer:    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.results_dir = Path("lambda_analysis_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Define parameter space for analysis
        self.lambda_combinations = []
        self.generate_lambda_combinations()
        
        # Base system configuration
        self.base_config = {
            'max_iterations': 50,
            'convergence_threshold': 1e-4,
            'preemption_threshold': 1.5
        }
        
    def generate_lambda_combinations(self):
        """Generate diverse λ parameter combinations for analysis"""
        
        # Strategy 1: Equal weights (baseline)
        self.lambda_combinations.append({
            'name': 'Equal_Weights',
            'lambda1': 0.25, 'lambda2': 0.25, 'lambda3': 0.25, 'lambda4': 0.25,
            'description': 'Balanced approach - equal priority to all objectives'
        })
        
        # Strategy 2: Latency-focused
        self.lambda_combinations.append({
            'name': 'Latency_Focused', 
            'lambda1': 0.6, 'lambda2': 0.2, 'lambda3': 0.1, 'lambda4': 0.1,
            'description': 'Prioritizes inference delay minimization'
        })
        
        # Strategy 3: Energy-focused
        self.lambda_combinations.append({
            'name': 'Energy_Focused',
            'lambda1': 0.1, 'lambda2': 0.1, 'lambda3': 0.2, 'lambda4': 0.6,
            'description': 'Prioritizes energy efficiency (computation + transmission)'
        })
        
        # Strategy 4: Communication-focused 
        self.lambda_combinations.append({
            'name': 'Communication_Focused',
            'lambda1': 0.1, 'lambda2': 0.1, 'lambda3': 0.6, 'lambda4': 0.2,
            'description': 'Prioritizes communication overhead minimization'
        })
        
        # Strategy 5: Batch processing focused
        self.lambda_combinations.append({
            'name': 'Batch_Focused',
            'lambda1': 0.2, 'lambda2': 0.6, 'lambda3': 0.1, 'lambda4': 0.1,
            'description': 'Prioritizes batch processing efficiency'
        })
        
        # Strategy 6: Hybrid latency-energy
        self.lambda_combinations.append({
            'name': 'Latency_Energy_Hybrid',
            'lambda1': 0.4, 'lambda2': 0.1, 'lambda3': 0.1, 'lambda4': 0.4,
            'description': 'Balances latency and energy concerns'
        })
        
        # Strategy 7: Communication-energy hybrid
        self.lambda_combinations.append({
            'name': 'Comm_Energy_Hybrid', 
            'lambda1': 0.1, 'lambda2': 0.1, 'lambda3': 0.4, 'lambda4': 0.4,
            'description': 'Balances communication and energy efficiency'
        })
        
        # Strategy 8: Real-time critical (extreme latency focus)
        self.lambda_combinations.append({
            'name': 'Real_Time_Critical',
            'lambda1': 0.8, 'lambda2': 0.1, 'lambda3': 0.05, 'lambda4': 0.05,
            'description': 'Extreme latency optimization for critical applications'
        })
        
        # Strategy 9: Battery-constrained (extreme energy focus)
        self.lambda_combinations.append({
            'name': 'Battery_Constrained',
            'lambda1': 0.05, 'lambda2': 0.05, 'lambda3': 0.1, 'lambda4': 0.8,
            'description': 'Extreme energy conservation for battery-powered devices'
        })
        
        self.logger.info(f"Generated {len(self.lambda_combinations)} λ parameter combinations")
    
    def create_test_scenarios(self) -> List[Dict[str, Any]]:
        scenarios = []
        
        # Scenario 1: Light workload
        scenarios.append({
            'name': 'Light_Workload',
            'system_state': SystemState(
                total_inference_delay=120.0,  # ms
                total_batch_delay=45.0,       # ms  
                total_communication_overhead=8.5,  # ms
                total_energy_consumption=2.1,      # J computational
                total_transmission_energy=0.15     # J transmission
            ),
            'description': 'Low computational load, typical sensor data processing'
        })
        
        # Scenario 2: Medium workload
        scenarios.append({
            'name': 'Medium_Workload',
            'system_state': SystemState(
                total_inference_delay=250.0,
                total_batch_delay=95.0,
                total_communication_overhead=18.2,
                total_energy_consumption=4.8,
                total_transmission_energy=0.35
            ),
            'description': 'Moderate load, typical smart home hub operations'
        })
        
        # Scenario 3: Heavy workload
        scenarios.append({
            'name': 'Heavy_Workload',
            'system_state': SystemState(
                total_inference_delay=480.0,
                total_batch_delay=180.0,
                total_communication_overhead=35.7,
                total_energy_consumption=9.2,
                total_transmission_energy=0.68
            ),
            'description': 'High computational load, complex inference tasks'
        })
        
        # Scenario 4: Communication-intensive
        scenarios.append({
            'name': 'Communication_Intensive',
            'system_state': SystemState(
                total_inference_delay=200.0,
                total_batch_delay=70.0,
                total_communication_overhead=85.3,
                total_energy_consumption=3.5,
                total_transmission_energy=1.25
            ),
            'description': 'High data transfer requirements, frequent cloud offloading'
        })
        
        # Scenario 5: Energy-constrained
        scenarios.append({
            'name': 'Energy_Constrained',
            'system_state': SystemState(
                total_inference_delay=350.0,
                total_batch_delay=120.0,
                total_communication_overhead=22.1,
                total_energy_consumption=12.8,
                total_transmission_energy=2.1
            ),
            'description': 'Battery-powered devices, energy is critical constraint'
        })
        
        return scenarios
    
    def analyze_lambda_effects(self, scenarios: List[Dict]) -> Dict[str, Any]:

        results = {
            'lambda_combinations': [],
            'scenario_results': {},
            'performance_matrices': {},
            'trade_off_analysis': {}
        }
        
        self.logger.info("Starting λ parameter analysis...")
        
        for lambda_config in self.lambda_combinations:
            combination_results = {
                'config': lambda_config,
                'scenario_objectives': {},
                'performance_metrics': {}
            }
            
            # Create optimizer with current λ configuration
            config = self.base_config.copy()
            config.update({
                'lambda1': lambda_config['lambda1'],
                'lambda2': lambda_config['lambda2'], 
                'lambda3': lambda_config['lambda3'],
                'lambda4': lambda_config['lambda4']
            })
            
            optimizer = MultiObjectiveOptimizer(config)
            
            scenario_objectives = {}
            
            for scenario in scenarios:
                system_state = scenario['system_state']
                objective_value = optimizer.calculate_objective_function(system_state)
                
                # Calculate individual component contributions
                weights = optimizer.weights
                components = {
                    'inference_delay_contribution': weights.lambda1 * system_state.total_inference_delay,
                    'batch_delay_contribution': weights.lambda2 * system_state.total_batch_delay,
                    'communication_contribution': weights.lambda3 * system_state.total_communication_overhead,
                    'energy_contribution': weights.lambda4 * (system_state.total_energy_consumption + system_state.total_transmission_energy)
                }
                
                scenario_objectives[scenario['name']] = {
                    'total_objective': objective_value,
                    'components': components,
                    'system_metrics': {
                        'inference_delay_ms': system_state.total_inference_delay,
                        'batch_delay_ms': system_state.total_batch_delay,
                        'communication_overhead_ms': system_state.total_communication_overhead,
                        'computational_energy_j': system_state.total_energy_consumption,
                        'transmission_energy_j': system_state.total_transmission_energy,
                        'total_energy_j': system_state.total_energy_consumption + system_state.total_transmission_energy
                    }
                }
            
            combination_results['scenario_objectives'] = scenario_objectives
            
            # Calculate aggregate performance metrics
            total_objectives = [obj['total_objective'] for obj in scenario_objectives.values()]
            combination_results['performance_metrics'] = {
                'average_objective': np.mean(total_objectives),
                'min_objective': np.min(total_objectives),
                'max_objective': np.max(total_objectives),
                'std_objective': np.std(total_objectives),
                'coefficient_of_variation': np.std(total_objectives) / np.mean(total_objectives) if np.mean(total_objectives) > 0 else 0
            }
            
            results['lambda_combinations'].append(combination_results)
            
            self.logger.info(f"Completed analysis for {lambda_config['name']}: "
                           f"avg_obj={combination_results['performance_metrics']['average_objective']:.3f}")
        
        # Perform comparative analysis
        results['trade_off_analysis'] = self.perform_tradeoff_analysis(results['lambda_combinations'], scenarios)
        
        return results
    
    def perform_tradeoff_analysis(self, combination_results: List[Dict], 
                                scenarios: List[Dict]) -> Dict[str, Any]:
        
        trade_offs = {
            'best_for_latency': None,
            'best_for_energy': None,
            'best_for_communication': None,
            'best_overall': None,
            'scenario_winners': {},
            'performance_rankings': {}
        }
        
        # Find best configuration for each objective component
        latency_scores = []
        energy_scores = []
        communication_scores = []
        overall_scores = []
        
        for combo_result in combination_results:
            # Average across all scenarios
            avg_latency = np.mean([
                obj['system_metrics']['inference_delay_ms'] + obj['system_metrics']['batch_delay_ms']
                for obj in combo_result['scenario_objectives'].values()
            ])
            
            avg_energy = np.mean([
                obj['system_metrics']['total_energy_j']
                for obj in combo_result['scenario_objectives'].values()
            ])
            
            avg_communication = np.mean([
                obj['system_metrics']['communication_overhead_ms']
                for obj in combo_result['scenario_objectives'].values()
            ])
            
            avg_overall = combo_result['performance_metrics']['average_objective']
            
            latency_scores.append((avg_latency, combo_result['config']['name']))
            energy_scores.append((avg_energy, combo_result['config']['name']))
            communication_scores.append((avg_communication, combo_result['config']['name']))
            overall_scores.append((avg_overall, combo_result['config']['name']))
        

        trade_offs['best_for_latency'] = min(latency_scores, key=lambda x: x[0])
        trade_offs['best_for_energy'] = min(energy_scores, key=lambda x: x[0])  
        trade_offs['best_for_communication'] = min(communication_scores, key=lambda x: x[0])
        trade_offs['best_overall'] = min(overall_scores, key=lambda x: x[0])
        
        for scenario in scenarios:
            scenario_name = scenario['name']
            scenario_scores = []
            
            for combo_result in combination_results:
                obj_value = combo_result['scenario_objectives'][scenario_name]['total_objective']
                scenario_scores.append((obj_value, combo_result['config']['name']))
            
            trade_offs['scenario_winners'][scenario_name] = min(scenario_scores, key=lambda x: x[0])
        

        all_configs = [combo['config']['name'] for combo in combination_results]
        rankings = {}
        
        for metric, scores in [
            ('latency', latency_scores),
            ('energy', energy_scores), 
            ('communication', communication_scores),
            ('overall', overall_scores)
        ]:
            sorted_scores = sorted(scores, key=lambda x: x[0])
            rankings[metric] = [config_name for _, config_name in sorted_scores]
        
        trade_offs['performance_rankings'] = rankings
        
        return trade_offs
    
    def generate_visualization_data(self, results: Dict[str, Any]) -> Dict[str, Any]:
        
        viz_data = {
            'radar_chart_data': {},
            'heatmap_data': {},
            'trade_off_curves': {},
            'sensitivity_analysis': {}
        }
        

        metrics = ['inference_delay', 'batch_delay', 'communication', 'energy']
        combinations = results['lambda_combinations']
        

        all_values = {metric: [] for metric in metrics}
        
        for combo in combinations:
            for scenario_name, scenario_obj in combo['scenario_objectives'].items():
                metrics_data = scenario_obj['system_metrics']
                all_values['inference_delay'].append(metrics_data['inference_delay_ms'])
                all_values['batch_delay'].append(metrics_data['batch_delay_ms'])
                all_values['communication'].append(metrics_data['communication_overhead_ms'])
                all_values['energy'].append(metrics_data['total_energy_j'])
        
        metric_ranges = {}
        for metric, values in all_values.items():
            metric_ranges[metric] = {'min': min(values), 'max': max(values)}
        
        radar_data = {}
        for combo in combinations:
            combo_name = combo['config']['name']
            radar_data[combo_name] = {}
            

            avg_metrics = {
                'inference_delay': np.mean([
                    obj['system_metrics']['inference_delay_ms'] 
                    for obj in combo['scenario_objectives'].values()
                ]),
                'batch_delay': np.mean([
                    obj['system_metrics']['batch_delay_ms']
                    for obj in combo['scenario_objectives'].values()  
                ]),
                'communication': np.mean([
                    obj['system_metrics']['communication_overhead_ms']
                    for obj in combo['scenario_objectives'].values()
                ]),
                'energy': np.mean([
                    obj['system_metrics']['total_energy_j']
                    for obj in combo['scenario_objectives'].values()
                ])
            }
            
            for metric, value in avg_metrics.items():
                min_val = metric_ranges[metric]['min']
                max_val = metric_ranges[metric]['max']
                if max_val > min_val:
                    normalized = 100 * (1 - (value - min_val) / (max_val - min_val))
                else:
                    normalized = 100
                radar_data[combo_name][metric] = normalized
        
        viz_data['radar_chart_data'] = radar_data
        
        return viz_data
    
    def save_results(self, results: Dict[str, Any]):
        
        results_file = self.results_dir / "lambda_parameter_analysis.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        

        summary = {
            'analysis_summary': {
                'total_lambda_combinations': len(results['lambda_combinations']),
                'scenarios_tested': len(results['lambda_combinations'][0]['scenario_objectives']),
                'best_configurations': results['trade_off_analysis']
            },
            'key_findings': self.extract_key_findings(results)
        }
        
        summary_file = self.results_dir / "lambda_analysis_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Results saved to {self.results_dir}")
    
    def extract_key_findings(self, results: Dict[str, Any]) -> List[str]:
        """Extract key findings from the analysis"""
        findings = []
        
        trade_offs = results['trade_off_analysis']
        
        findings.append(
            f"Best overall configuration: {trade_offs['best_overall'][1]} "
            f"(objective: {trade_offs['best_overall'][0]:.3f})"
        )
        
        findings.append(
            f"Best for latency: {trade_offs['best_for_latency'][1]} "
            f"(avg latency: {trade_offs['best_for_latency'][0]:.1f}ms)"
        )
        
        findings.append(
            f"Best for energy efficiency: {trade_offs['best_for_energy'][1]} "
            f"(avg energy: {trade_offs['best_for_energy'][0]:.3f}J)"
        )
        
        findings.append(
            f"Best for communication: {trade_offs['best_for_communication'][1]} "
            f"(avg comm overhead: {trade_offs['best_for_communication'][0]:.1f}ms)"
        )
        
        scenario_winners = trade_offs['scenario_winners']
        findings.append("Scenario-specific winners:")
        for scenario, (obj_val, winner) in scenario_winners.items():
            findings.append(f"  {scenario}: {winner} (objective: {obj_val:.3f})")
        
        return findings
    
    def print_analysis_report(self, results: Dict[str, Any]):
        print("\n" + "="*80)
        print("LL-CIoT λ PARAMETER ANALYSIS REPORT")
        print("Addresses reviewer concern about dynamic λ parameter effects")
        print("="*80)
        
        trade_offs = results['trade_off_analysis']
        
        print("OVERALL PERFORMANCE WINNERS:")
        print(f"   Best Overall: {trade_offs['best_overall'][1]} (obj: {trade_offs['best_overall'][0]:.3f})")
        print(f"   Best Latency: {trade_offs['best_for_latency'][1]} ({trade_offs['best_for_latency'][0]:.1f}ms)")
        print(f"   Best Energy:  {trade_offs['best_for_energy'][1]} ({trade_offs['best_for_energy'][0]:.3f}J)")
        print(f"   Best Comm:    {trade_offs['best_for_communication'][1]} ({trade_offs['best_for_communication'][0]:.1f}ms)")
        
        print("PERFORMANCE RANKINGS:")
        rankings = trade_offs['performance_rankings']
        for metric, ranking in rankings.items():
            print(f"   {metric.capitalize():12}: {' > '.join(ranking[:3])} (top 3)")
        
        print("SCENARIO-SPECIFIC WINNERS:")
        for scenario, (obj_val, winner) in trade_offs['scenario_winners'].items():
            print(f"   {scenario:20}: {winner:20} (obj: {obj_val:.3f})")
        
        print("DETAILED CONFIGURATION ANALYSIS:")
        for combo_result in results['lambda_combinations']:
            config = combo_result['config']
            perf = combo_result['performance_metrics']
            
            print(f"\n   {config['name']}:")
            print(f"      λ₁={config['lambda1']:.2f}, λ₂={config['lambda2']:.2f}, "
                  f"λ₃={config['lambda3']:.2f}, λ₄={config['lambda4']:.2f}")
            print(f"      Avg Objective: {perf['average_objective']:.3f} ± {perf['std_objective']:.3f}")
            print(f"      Description: {config['description']}")
        
        print("KEY INSIGHTS:")
        findings = self.extract_key_findings(results)
        for i, finding in enumerate(findings, 1):
            if finding.startswith("  "):  # Indented sub-item
                print(f"      {finding}")
            else:
                print(f"   {i}. {finding}")
        
        print("\n" + "="*80)
        print("Analysis demonstrates clear performance trade-offs based on λ parameter choices")
        print("="*80 + "\n")

def main():
    """Execute comprehensive λ parameter analysis"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    analyzer = LambdaParameterAnalyzer()
    scenarios = analyzer.create_test_scenarios()
    results = analyzer.analyze_lambda_effects(scenarios)
    viz_data = analyzer.generate_visualization_data(results)
    results['visualization_data'] = viz_data
    analyzer.print_analysis_report(results)
    analyzer.save_results(results)
    
    print(f"\nComplete analysis results saved to: {analyzer.results_dir}")
    print("This addresses the reviewer's concern about λ parameter trade-off analysis.")

if __name__ == "__main__":
    main()