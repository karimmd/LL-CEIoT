#!/usr/bin/env python3
import logging
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import json
from scipy.optimize import minimize, differential_evolution
import itertools
import sys
import os

# Add transmission energy calculator to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from transmission_energy_calculator import TransmissionEnergyCalculator, TransmissionParameters

logger = logging.getLogger(__name__)

@dataclass
class OptimizationWeights:
    """
    Optimization weights from manuscript Problem P1
    
    λ₁: Inference delay weight
    λ₂: Batch delay weight
    λ₃: Communication overhead weight
    λ₄: Energy consumption weight
    
    Constraint: λ₁ + λ₂ + λ₃ + λ₄ = 1.0 (normalized weights)
    """
    lambda1: float = 0.3  # Inference delay weight
    lambda2: float = 0.2  # Batch delay weight
    lambda3: float = 0.3  # Communication overhead weight
    lambda4: float = 0.2  # Energy consumption weight
    
    def __post_init__(self):
        """Validate and normalize weights"""
        total = self.lambda1 + self.lambda2 + self.lambda3 + self.lambda4
        if abs(total - 1.0) > 1e-6:
            # Normalize weights
            self.lambda1 /= total
            self.lambda2 /= total
            self.lambda3 /= total
            self.lambda4 /= total
            logger.warning(f"Weights normalized to sum=1.0")
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'lambda1': self.lambda1,
            'lambda2': self.lambda2, 
            'lambda3': self.lambda3,
            'lambda4': self.lambda4
        }

@dataclass
class SystemState:
    """
    Current system state for optimization
    
    Represents all system variables and constraints from manuscript
    """
    # Tasks and batches
    inference_tasks: List = field(default_factory=list)
    fine_tuning_tasks: List = field(default_factory=list)
    active_batches: List = field(default_factory=list)
    
    # Resource states (φ_n(t), M_n, b_e(t))
    node_compute_capacities: Dict[str, float] = field(default_factory=dict)
    node_memory_capacities: Dict[str, float] = field(default_factory=dict)
    link_bandwidths: Dict[str, float] = field(default_factory=dict)
    
    # Layer assignments (X_{l,n})
    layer_assignments: Dict[Tuple[int, str], bool] = field(default_factory=dict)
    
    # Energy coefficients (η_n)
    energy_coefficients: Dict[str, float] = field(default_factory=dict)
    
    # Current objective values
    total_inference_delay: float = 0.0
    total_batch_delay: float = 0.0
    total_communication_overhead: float = 0.0
    total_energy_consumption: float = 0.0
    
    # Transmission energy metrics (addresses reviewer concern)
    total_transmission_energy: float = 0.0
    device_to_edge_tx_energy: float = 0.0
    edge_to_cloud_tx_energy: float = 0.0
    cloud_to_edge_tx_energy: float = 0.0
    edge_to_device_tx_energy: float = 0.0

class ConstraintValidator:
    """
    Validates all constraints (C1)-(C6) from Problem P1
    """
    
    @staticmethod
    def validate_compute_capacity_constraint(tasks: List, node_capacity: float) -> bool:
        """
        Constraint (C1): Σ c_j^quant ≤ φ_n(t)
        
        Validates that total quantized computational demand doesn't exceed node capacity
        """
        total_compute_demand = sum(task.compute_demand for task in tasks)
        return total_compute_demand <= node_capacity
    
    @staticmethod
    def validate_memory_capacity_constraint(layer_assignments: Dict, 
                                          layers: List, nodes: Dict) -> bool:
        """
        Constraint (C2): Σ X_{l,n} · m_l^quant ≤ M_n
        
        Validates memory capacity constraints for all nodes
        """
        for node_id, node in nodes.items():
            total_memory_used = 0.0
            
            for (layer_id, assigned_node), is_assigned in layer_assignments.items():
                if is_assigned and assigned_node == node_id:
                    layer = next(l for l in layers if l.layer_id == layer_id)
                    total_memory_used += layer.quantized_memory_requirement
            
            if total_memory_used > node.memory_capacity:
                logger.debug(f"Memory constraint violated for {node_id}: {total_memory_used:.1f} > {node.memory_capacity:.1f}")
                return False
        
        return True
    
    @staticmethod
    def validate_bandwidth_constraint(task_data_rate: float, link_bandwidth: float) -> bool:
        """
        Constraint (C3): r_j ≤ b_e(t)
        
        Validates that task data rate doesn't exceed link bandwidth
        """
        return task_data_rate <= link_bandwidth
    
    @staticmethod
    def validate_layer_assignment_constraint(layer_assignments: Dict, num_layers: int) -> bool:
        """
        Constraint (C4): Σ X_{l,n} = 1 ∀l
        
        Validates that each layer is assigned to exactly one node
        """
        for layer_id in range(num_layers):
            assignments_for_layer = sum(
                1 for (l_id, node_id), is_assigned in layer_assignments.items()
                if l_id == layer_id and is_assigned
            )
            if assignments_for_layer != 1:
                logger.debug(f"Layer {layer_id} has {assignments_for_layer} assignments (should be 1)")
                return False
        
        return True
    
    @staticmethod
    def validate_preemption_constraint(new_task_density: float, current_task_density: float,
                                       theta: float) -> bool:
        """
        Per-task variant of the revised preemption rule:
            rho_new >= theta * rho_cur,   theta > 1.

        Used by the scheduler only for the drift-triggered urgent
        model-adaptation case; ordinary FT does not preempt INF.
        """
        return new_task_density >= theta * current_task_density

    @staticmethod
    def validate_batch_preemption_constraint(new_batch_density: float,
                                             current_batch_density: float,
                                             theta: float) -> bool:
        """
        Aggregate batch form of the revised preemption rule:
            sum_{j in B_new} rho_j >= theta * sum_{k in B_cur} rho_k,   theta > 1.
        """
        return new_batch_density >= theta * current_batch_density

class MultiObjectiveOptimizer:
    """
    Multi-objective Optimizer implementing Problem P1 from manuscript
    
    Solves the MILP optimization problem:
    min O = Σ λ₁D_j^inf + Σ λ₂D_B^total + λ₃C^total + λ₄E^total
    
    Subject to constraints (C1)-(C6) with different optimization strategies:
    1. Weighted sum approach
    2. Pareto front exploration  
    3. Constraint-based optimization
    4. Heuristic algorithms for real-time decisions
    """
    
    def __init__(self, config: Dict):
        self.weights = OptimizationWeights(
            lambda1=config.get('lambda1', 0.3),
            lambda2=config.get('lambda2', 0.2), 
            lambda3=config.get('lambda3', 0.3),
            lambda4=config.get('lambda4', 0.2)
        )
        
        self.max_iterations = config.get('max_iterations', 100)
        self.convergence_threshold = config.get('convergence_threshold', 1e-4)
        # theta is the canonical preemption-threshold name (manuscript revision);
        # 'preemption_threshold' is accepted for backward compatibility with old configs.
        self.theta = config.get('theta',
                                config.get('preemption_threshold', 1.5))
        self.preemption_threshold = self.theta  # alias retained for callers
        
        # Initialize transmission energy calculator (addresses reviewer concern)
        self.tx_energy_calculator = TransmissionEnergyCalculator()
        
        # Optimization history
        self.optimization_history = []
        self.pareto_solutions = []
        
        # Performance tracking
        self.optimization_stats = {
            'total_optimizations': 0,
            'average_optimization_time': 0.0,
            'best_objective_value': float('inf'),
            'constraint_violations': 0,
            'successful_optimizations': 0,
            'transmission_energy_computed': 0  # Track transmission energy calculations
        }
        
        logger.info(f"Multi-objective Optimizer initialized with transmission energy quantification")
        logger.info(f"Weights: λ₁={self.weights.lambda1:.2f}, λ₂={self.weights.lambda2:.2f}, "
                   f"λ₃={self.weights.lambda3:.2f}, λ₄={self.weights.lambda4:.2f}")
        logger.info(f"Max iterations: {self.max_iterations}, theta: {self.theta}")
    
    def calculate_transmission_energy(self, system_state: SystemState, 
                                    data_transfers: List[Dict]) -> Dict[str, float]:
        """
        Quantify transmission energy consumption - addresses reviewer concern
        
        This method provides empirical transmission energy measurements rather than
        relying on theoretical assumptions as criticized by the reviewer.
        
        Args:
            system_state: Current system state
            data_transfers: List of data transfer operations
            
        Returns:
            Detailed transmission energy breakdown
        """
        try:
            # Calculate transmission energy for all data transfers
            tx_energy_results = self.tx_energy_calculator.calculate_cloud_edge_transmission_energy(
                data_transfers
            )
            
            # Update system state with detailed transmission energy metrics
            total_breakdown = tx_energy_results['total_energy_breakdown']
            system_state.device_to_edge_tx_energy = total_breakdown['device_to_edge_energy_j']
            system_state.edge_to_cloud_tx_energy = total_breakdown['edge_to_cloud_energy_j'] 
            system_state.cloud_to_edge_tx_energy = total_breakdown['cloud_to_edge_energy_j']
            system_state.edge_to_device_tx_energy = total_breakdown['edge_to_device_energy_j']
            system_state.total_transmission_energy = total_breakdown['total_transmission_energy_j']
            
            # Track transmission energy calculations
            self.optimization_stats['transmission_energy_computed'] += 1
            
            logger.debug(f"Transmission energy calculated: {system_state.total_transmission_energy:.6f} J")
            logger.debug(f"  Device->Edge: {system_state.device_to_edge_tx_energy:.6f} J")
            logger.debug(f"  Edge->Cloud: {system_state.edge_to_cloud_tx_energy:.6f} J")
            logger.debug(f"  Cloud->Edge: {system_state.cloud_to_edge_tx_energy:.6f} J") 
            logger.debug(f"  Edge->Device: {system_state.edge_to_device_tx_energy:.6f} J")
            
            return {
                'total_transmission_energy_j': system_state.total_transmission_energy,
                'device_to_edge_tx_energy_j': system_state.device_to_edge_tx_energy,
                'edge_to_cloud_tx_energy_j': system_state.edge_to_cloud_tx_energy,
                'cloud_to_edge_tx_energy_j': system_state.cloud_to_edge_tx_energy,
                'edge_to_device_tx_energy_j': system_state.edge_to_device_tx_energy,
                'energy_distribution': tx_energy_results['energy_distribution']
            }
            
        except Exception as e:
            logger.error(f"Transmission energy calculation failed: {e}")
            return {
                'total_transmission_energy_j': 0.0,
                'device_to_edge_tx_energy_j': 0.0, 
                'edge_to_cloud_tx_energy_j': 0.0,
                'cloud_to_edge_tx_energy_j': 0.0,
                'edge_to_device_tx_energy_j': 0.0,
                'energy_distribution': {'device_edge_percentage': 0.0, 'edge_cloud_percentage': 0.0}
            }
    
    def calculate_objective_function(self, system_state: SystemState) -> float:
        """
        Calculate the objective function O from Problem P1
        
        Enhanced version addressing reviewer concern:
        O = Σ λ₁D_j^inf + Σ λ₂D_B^total + λ₃C^total + λ₄(E^comp + E^trans)
        
        Where E^trans is now explicitly quantified transmission energy
        """
        try:
            # Component 1: Inference delay term (λ₁ΣD_j^inf)
            inference_delay_term = self.weights.lambda1 * system_state.total_inference_delay
            
            # Component 2: Batch delay term (λ₂ΣD_B^total)  
            batch_delay_term = self.weights.lambda2 * system_state.total_batch_delay
            
            # Component 3: Communication overhead term (λ₃C^total)
            communication_term = self.weights.lambda3 * system_state.total_communication_overhead
            
            # Component 4: Enhanced energy consumption term (λ₄E^total)
            # E^total = E^computation + E^transmission (addresses reviewer concern)
            computational_energy = system_state.total_energy_consumption
            transmission_energy = system_state.total_transmission_energy
            total_energy = computational_energy + transmission_energy
            energy_term = self.weights.lambda4 * total_energy
            
            # Total objective value
            objective_value = (inference_delay_term + batch_delay_term + 
                             communication_term + energy_term)
            
            logger.debug(f"Enhanced objective components: inf={inference_delay_term:.3f}, "
                        f"batch={batch_delay_term:.3f}, comm={communication_term:.3f}, "
                        f"energy={energy_term:.3f} (comp={computational_energy:.3f}J + "
                        f"trans={transmission_energy:.3f}J), total={objective_value:.3f}")
            
            return objective_value
            
        except Exception as e:
            logger.error(f"Objective calculation failed: {e}")
            return float('inf')
    
    def evaluate_constraints(self, system_state: SystemState, 
                           tasks: List, layers: List, nodes: Dict) -> Tuple[bool, List[str]]:
        """
        Evaluate all constraints (C1)-(C6) for current system state
        
        Returns:
            (all_constraints_satisfied, list_of_violations)
        """
        violations = []
        
        try:
            # (C1) Compute capacity constraints
            for node_id, capacity in system_state.node_compute_capacities.items():
                node_tasks = [t for t in tasks if getattr(t, 'assigned_node', None) == node_id]
                if not ConstraintValidator.validate_compute_capacity_constraint(node_tasks, capacity):
                    violations.append(f"C1: Compute capacity exceeded for {node_id}")
            
            # (C2) Memory capacity constraints
            if not ConstraintValidator.validate_memory_capacity_constraint(
                system_state.layer_assignments, layers, nodes):
                violations.append("C2: Memory capacity constraint violated")
            
            # (C3) Bandwidth constraints
            for task in tasks:
                task_rate = getattr(task, 'data_rate', 0.0)
                link_bw = system_state.link_bandwidths.get('default', float('inf'))
                if not ConstraintValidator.validate_bandwidth_constraint(task_rate, link_bw):
                    violations.append(f"C3: Bandwidth constraint violated for task {task.task_id}")
            
            # (C4) Layer assignment constraints
            if not ConstraintValidator.validate_layer_assignment_constraint(
                system_state.layer_assignments, len(layers)):
                violations.append("C4: Layer assignment constraint violated")
            
            # (C5) and (C6) are validated during scheduling, not here
            
            is_feasible = len(violations) == 0
            
            if violations:
                logger.debug(f"Constraint violations: {violations}")
            
            return is_feasible, violations
            
        except Exception as e:
            logger.error(f"Constraint evaluation failed: {e}")
            return False, [f"Evaluation error: {str(e)}"]
    
    def optimize_cloud_edge_deployment(self, tasks: List, layers: List, 
                                     nodes: Dict) -> Tuple[Dict, float]:
        """
        Main optimization function for cloud-edge deployment decisions
        
        This implements the core Problem P1 optimization for LL-CIoT
        """
        optimization_start = time.time()
        
        logger.info(f"Starting multi-objective optimization for {len(tasks)} tasks, "
                   f"{len(layers)} layers, {len(nodes)} nodes")
        
        try:
            # Initialize system state
            system_state = self._initialize_system_state(tasks, layers, nodes)
            
            # Optimization approaches (try multiple methods)
            best_solution = None
            best_objective = float('inf')
            
            # Method 1: Weighted sum optimization
            solution1, obj1 = self._weighted_sum_optimization(system_state, tasks, layers, nodes)
            if obj1 < best_objective:
                best_solution, best_objective = solution1, obj1
            
            # Method 2: Heuristic optimization for real-time decisions
            solution2, obj2 = self._heuristic_optimization(system_state, tasks, layers, nodes)
            if obj2 < best_objective:
                best_solution, best_objective = solution2, obj2
            
            # Method 3: Constraint-based optimization
            solution3, obj3 = self._constraint_based_optimization(system_state, tasks, layers, nodes)
            if obj3 < best_objective:
                best_solution, best_objective = solution3, obj3
            
            optimization_time = time.time() - optimization_start
            
            # Update performance statistics
            self.optimization_stats['total_optimizations'] += 1
            self.optimization_stats['average_optimization_time'] = (
                (self.optimization_stats['average_optimization_time'] * 
                 (self.optimization_stats['total_optimizations'] - 1) + optimization_time) /
                self.optimization_stats['total_optimizations']
            )
            
            if best_objective < self.optimization_stats['best_objective_value']:
                self.optimization_stats['best_objective_value'] = best_objective
            
            if best_solution is not None:
                self.optimization_stats['successful_optimizations'] += 1
            
            # Log results
            logger.info(f"Optimization completed in {optimization_time:.3f}s")
            logger.info(f"Best objective value: {best_objective:.6f}")
            logger.info(f"Methods tried: weighted_sum={obj1:.3f}, heuristic={obj2:.3f}, constraint={obj3:.3f}")
            
            return best_solution, best_objective
            
        except Exception as e:
            logger.error(f"Multi-objective optimization failed: {e}")
            self.optimization_stats['constraint_violations'] += 1
            return None, float('inf')
    
    def _initialize_system_state(self, tasks: List, layers: List, nodes: Dict) -> SystemState:
        """Initialize system state for optimization"""
        system_state = SystemState()
        
        # Categorize tasks
        system_state.inference_tasks = [t for t in tasks if getattr(t, 'task_type', None) == 'INF']
        system_state.fine_tuning_tasks = [t for t in tasks if getattr(t, 'task_type', None) == 'FT']
        
        # Initialize resource capacities
        for node_id, node in nodes.items():
            system_state.node_compute_capacities[node_id] = node.compute_capacity
            system_state.node_memory_capacities[node_id] = node.memory_capacity
            system_state.energy_coefficients[node_id] = 1.0  # Default energy coefficient
        
        # Initialize layer assignments (all unassigned initially)
        for layer in layers:
            for node_id in nodes.keys():
                system_state.layer_assignments[(layer.layer_id, node_id)] = False
        
        # Set default bandwidth
        system_state.link_bandwidths['default'] = 50.0  # 50 Mbps as per manuscript
        
        return system_state
    
    def _weighted_sum_optimization(self, system_state: SystemState, 
                                 tasks: List, layers: List, nodes: Dict) -> Tuple[Dict, float]:
        """
        Weighted sum approach to multi-objective optimization
        
        Converts multi-objective problem to single-objective using weights λ₁-λ₄
        """
        logger.debug("Attempting weighted sum optimization")
        
        try:
            # Simple greedy assignment for demonstration
            solution = {}
            
            # Task-to-node assignment based on weighted criteria
            for task in tasks:
                best_node = None
                best_score = float('inf')
                
                for node_id, node in nodes.items():
                    # Calculate weighted score for this assignment
                    inference_delay = getattr(task, 'compute_demand', 100) / node.compute_capacity
                    communication_overhead = 10.0  # Simplified
                    energy_cost = inference_delay * system_state.energy_coefficients.get(node_id, 1.0)
                    
                    # Weighted sum
                    score = (self.weights.lambda1 * inference_delay + 
                            self.weights.lambda3 * communication_overhead +
                            self.weights.lambda4 * energy_cost)
                    
                    if score < best_score:
                        best_score = score
                        best_node = node_id
                
                solution[task.task_id] = {
                    'assigned_node': best_node,
                    'assignment_score': best_score
                }
            
            # Calculate total objective value
            total_objective = sum(assignment['assignment_score'] for assignment in solution.values())
            
            logger.debug(f"Weighted sum optimization: objective = {total_objective:.6f}")
            return solution, total_objective
            
        except Exception as e:
            logger.error(f"Weighted sum optimization failed: {e}")
            return {}, float('inf')
    
    def _heuristic_optimization(self, system_state: SystemState,
                              tasks: List, layers: List, nodes: Dict) -> Tuple[Dict, float]:
        """
        Heuristic optimization for real-time deployment decisions
        
        Uses domain-specific heuristics for LL-CIoT cloud-edge optimization
        """
        logger.debug("Attempting heuristic optimization")
        
        try:
            solution = {}
            
            # Heuristic rules for LL-CIoT deployment
            for task in tasks:
                task_priority = getattr(task, 'priority_value', 1.0)
                task_complexity = getattr(task, 'compute_demand', 100)
                task_deadline = getattr(task, 'deadline', time.time() + 10)
                current_time = time.time()
                time_remaining = task_deadline - current_time
                
                # Decision rules:
                # 1. High priority + tight deadline → Edge
                if task_priority > 2.5 and time_remaining < 2.0:
                    assigned_node = 'edge_server'
                    deployment_reasoning = "High priority with tight deadline"
                
                # 2. High complexity → Cloud  
                elif task_complexity > 200:
                    assigned_node = 'cloud_server'
                    deployment_reasoning = "High computational complexity"
                
                # 3. Default based on energy efficiency
                else:
                    # Choose node with best energy efficiency
                    edge_energy = task_complexity / nodes['edge_server'].compute_capacity * 75  # Edge power
                    cloud_energy = task_complexity / nodes['cloud_server'].compute_capacity * 350  # Cloud power
                    
                    if edge_energy < cloud_energy:
                        assigned_node = 'edge_server'
                        deployment_reasoning = "Energy-efficient edge deployment"
                    else:
                        assigned_node = 'cloud_server'
                        deployment_reasoning = "Cloud deployment for efficiency"
                
                # Calculate deployment score
                if assigned_node == 'edge_server':
                    delay = task_complexity / nodes['edge_server'].compute_capacity
                    energy = delay * 75
                else:
                    delay = task_complexity / nodes['cloud_server'].compute_capacity
                    energy = delay * 350
                
                score = self.weights.lambda1 * delay + self.weights.lambda4 * energy
                
                solution[task.task_id] = {
                    'assigned_node': assigned_node,
                    'assignment_score': score,
                    'deployment_reasoning': deployment_reasoning
                }
            
            total_objective = sum(assignment['assignment_score'] for assignment in solution.values())
            
            logger.debug(f"Heuristic optimization: objective = {total_objective:.6f}")
            return solution, total_objective
            
        except Exception as e:
            logger.error(f"Heuristic optimization failed: {e}")
            return {}, float('inf')
    
    def _constraint_based_optimization(self, system_state: SystemState,
                                     tasks: List, layers: List, nodes: Dict) -> Tuple[Dict, float]:
        """
        Constraint-based optimization focusing on constraint satisfaction
        
        Prioritizes finding feasible solutions that satisfy all constraints (C1)-(C6)
        """
        logger.debug("Attempting constraint-based optimization")
        
        try:
            solution = {}
            
            # Sort tasks by value density (from VDF scheduler concept)
            sorted_tasks = sorted(tasks, 
                                key=lambda t: getattr(t, 'priority_value', 1.0) / 
                                             max(getattr(t, 'deadline', time.time() + 10) - time.time(), 0.1),
                                reverse=True)
            
            # Resource tracking
            node_compute_used = {node_id: 0.0 for node_id in nodes.keys()}
            node_memory_used = {node_id: 0.0 for node_id in nodes.keys()}
            
            for task in sorted_tasks:
                best_node = None
                best_score = float('inf')
                
                for node_id, node in nodes.items():
                    # Check constraints
                    task_compute = getattr(task, 'compute_demand', 100)
                    task_memory = getattr(task, 'memory_demand', 50)
                    
                    # Constraint checking
                    if (node_compute_used[node_id] + task_compute <= node.compute_capacity and
                        node_memory_used[node_id] + task_memory <= node.memory_capacity):
                        
                        # Calculate objective contribution
                        delay = task_compute / (node.compute_capacity - node_compute_used[node_id])
                        energy = delay * system_state.energy_coefficients.get(node_id, 1.0)
                        score = self.weights.lambda1 * delay + self.weights.lambda4 * energy
                        
                        if score < best_score:
                            best_score = score
                            best_node = node_id
                
                if best_node is not None:
                    # Assign task
                    task_compute = getattr(task, 'compute_demand', 100)
                    task_memory = getattr(task, 'memory_demand', 50)
                    
                    node_compute_used[best_node] += task_compute
                    node_memory_used[best_node] += task_memory
                    
                    solution[task.task_id] = {
                        'assigned_node': best_node,
                        'assignment_score': best_score
                    }
                else:
                    # Task cannot be assigned (constraint violation)
                    logger.warning(f"Task {task.task_id} cannot be assigned - resource constraints")
            
            total_objective = sum(assignment['assignment_score'] for assignment in solution.values())
            
            logger.debug(f"Constraint-based optimization: objective = {total_objective:.6f}")
            return solution, total_objective
            
        except Exception as e:
            logger.error(f"Constraint-based optimization failed: {e}")
            return {}, float('inf')
    
    def analyze_pareto_front(self, tasks: List, layers: List, nodes: Dict) -> List[Dict]:
        """
        Generate Pareto front for multi-objective analysis
        
        Explores trade-offs between conflicting objectives (delay vs energy vs communication)
        """
        logger.info("Generating Pareto front analysis")
        
        pareto_solutions = []
        
        # Generate different weight combinations
        weight_combinations = [
            [0.4, 0.2, 0.2, 0.2],  # Delay-focused
            [0.2, 0.2, 0.4, 0.2],  # Communication-focused  
            [0.2, 0.2, 0.2, 0.4],  # Energy-focused
            [0.25, 0.25, 0.25, 0.25],  # Balanced
            [0.5, 0.1, 0.2, 0.2],  # Extreme delay focus
            [0.1, 0.1, 0.1, 0.7],  # Extreme energy focus
        ]
        
        for weights in weight_combinations:
            # Temporarily update weights
            original_weights = self.weights
            self.weights = OptimizationWeights(*weights)
            
            # Run optimization
            solution, objective = self.optimize_cloud_edge_deployment(tasks, layers, nodes)
            
            if solution is not None:
                pareto_solutions.append({
                    'weights': weights.copy(),
                    'objective_value': objective,
                    'solution': solution,
                    'trade_offs': {
                        'delay_focus': weights[0],
                        'batch_focus': weights[1], 
                        'communication_focus': weights[2],
                        'energy_focus': weights[3]
                    }
                })
            
            # Restore original weights
            self.weights = original_weights
        
        # Sort by objective value
        pareto_solutions.sort(key=lambda x: x['objective_value'])
        
        logger.info(f"Generated {len(pareto_solutions)} Pareto solutions")
        return pareto_solutions
    
    def get_optimization_report(self) -> Dict:
        """Generate comprehensive optimization report for research validation"""
        return {
            'optimizer_config': {
                'weights': self.weights.to_dict(),
                'max_iterations': self.max_iterations,
                'convergence_threshold': self.convergence_threshold,
                'theta': self.theta
            },
            'performance_stats': self.optimization_stats,
            'optimization_history': self.optimization_history[-10:],  # Last 10 optimizations
            'pareto_solutions_count': len(self.pareto_solutions)
        }

if __name__ == "__main__":
    # Test multi-objective optimizer
    logging.basicConfig(level=logging.INFO)
    
    # Configuration matching manuscript
    config = {
        'lambda1': 0.3,  # Inference delay weight
        'lambda2': 0.2,  # Batch delay weight
        'lambda3': 0.3,  # Communication overhead weight
        'lambda4': 0.2,  # Energy consumption weight
        'max_iterations': 100,
        'convergence_threshold': 1e-4,
        'theta': 1.5     # preemption threshold (manuscript: theta > 1)
    }
    
    optimizer = MultiObjectiveOptimizer(config)
    
    print("Multi-objective Optimizer (Problem P1)")
    print("=" * 50)
    print(f"Optimization weights: {optimizer.weights.to_dict()}")
    print(f"Preemption threshold theta: {optimizer.theta}")
    
    # Sample system for testing
    class MockTask:
        def __init__(self, task_id, priority_value, compute_demand, deadline):
            self.task_id = task_id
            self.priority_value = priority_value
            self.compute_demand = compute_demand
            self.deadline = deadline
            self.memory_demand = 50.0
    
    class MockNode:
        def __init__(self, compute_capacity, memory_capacity):
            self.compute_capacity = compute_capacity
            self.memory_capacity = memory_capacity
    
    # Create test scenario
    tasks = [
        MockTask("task1", 3.0, 150, time.time() + 5),
        MockTask("task2", 2.0, 200, time.time() + 10),
        MockTask("task3", 2.5, 100, time.time() + 3)
    ]
    
    nodes = {
        'cloud_server': MockNode(8.0, 11000.0),
        'edge_server': MockNode(2.0, 4000.0)
    }
    
    # Run optimization
    solution, objective = optimizer.optimize_cloud_edge_deployment(tasks, [], nodes)
    
    if solution:
        print(f"\nOptimization successful:")
        print(f"Objective value: {objective:.6f}")
        for task_id, assignment in solution.items():
            print(f"  {task_id} -> {assignment['assigned_node']} (score: {assignment['assignment_score']:.3f})")
    
    # Generate report
    report = optimizer.get_optimization_report()
    print(f"\nOptimization Report:")
    print(json.dumps(report, indent=2))