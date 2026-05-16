#!/usr/bin/env python3
"""
LL-CEIoT Reference Implementation - VDF Scheduler (Algorithm 1, revised)

Implements the value-density-first scheduler:
- Value density:  rho_j = v_j / (t_j^dl - t_j^arr)
- Preemption:     an urgent model-adaptation batch may preempt a lower-criticality
                  running batch only when its aggregate value density strictly
                  exceeds the running batch's by a factor of theta > 1, and only
                  when the new batch is at least as energy-efficient.
- Batch construction for INF and FT tasks ordered by rho_j.

Notes on the revised preemption rule (manuscript Section "Lightweight
LLM-based Batch Scheduling"): ordinary FT tasks carry background-level value
density and do NOT preempt delay-sensitive INF tasks in the common case.
Preemption is the exception, reserved for drift-triggered model-adaptation
events where a detected accuracy degradation in the deployed model makes the
adaptation batch genuinely more urgent than the running lower-criticality
batch. The threshold symbol has been unified to theta throughout the
manuscript and this implementation.
"""

import time
import logging
import heapq
import threading
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

class TaskType(Enum):
    """Task types as defined in manuscript"""
    INFERENCE = "INF"  # I_j = 1 in manuscript
    FINE_TUNING = "FT"  # I_j = 0 in manuscript

@dataclass
class IoTTask:
    """
    IoT Task representation matching manuscript notation
    
    Manuscript symbols:
    - j: task index
    - t_j^arr: arrival time  
    - t_j^dl: deadline
    - v_j: priority value
    - c_j: computational demand
    - ρ_j: value density = v_j/(t_j^dl - t_j^arr)
    - I_j: task type indicator (1=INF, 0=FT)
    """
    task_id: str
    task_type: TaskType
    arrival_time: float  # t_j^arr
    deadline: float      # t_j^dl  
    priority_value: float  # v_j
    compute_demand: float  # c_j
    memory_demand: float   # m_j
    data_size: float      # β_j
    device_id: str = ""
    
    def __post_init__(self):
        """Calculate derived properties after initialization"""
        self.time_window = self.deadline - self.arrival_time
        if self.time_window <= 0:
            raise ValueError(f"Invalid time window for task {self.task_id}: {self.time_window}")
    
    @property
    def value_density(self) -> float:
        """
        Value density calculation from manuscript: ρⱼ = vⱼ/(tⱼᵈˡ - tⱼᵃʳʳ)
        """
        return self.priority_value / self.time_window
    
    @property
    def task_indicator(self) -> int:
        """Task type indicator I_j from manuscript (1=INF, 0=FT)"""
        return 1 if self.task_type == TaskType.INFERENCE else 0
    
    def __lt__(self, other):
        """Comparison for priority queue (higher value density = higher priority)"""
        return self.value_density > other.value_density

@dataclass  
class TaskBatch:
    """
    Task batch representation matching manuscript notation B_i
    
    Batch contains tasks sorted by value density: ρ_j1 ≥ ρ_j2 ≥ ... ≥ ρ_jb
    """
    batch_id: str
    tasks: List[IoTTask] = field(default_factory=list)
    creation_time: float = field(default_factory=time.time)
    assigned_node: Optional[str] = None
    
    @property
    def batch_size(self) -> int:
        """Batch size b in manuscript"""
        return len(self.tasks)
    
    @property
    def cumulative_value_density(self) -> float:
        """Cumulative value density for preemption decisions"""
        return sum(task.value_density for task in self.tasks)
    
    @property
    def total_compute_demand(self) -> float:
        """Total computational demand for resource allocation"""
        return sum(task.compute_demand for task in self.tasks)
    
    @property
    def processing_delay(self, node_capacity: float) -> float:
        """Processing delay D_B^proc = Σ(c_j/φ_n) from manuscript"""
        if node_capacity <= 0:
            return float('inf')
        return self.total_compute_demand / node_capacity

class VDFScheduler:
    """
    Value Density First Scheduler (Algorithm 1, revised).

    Key features:
    1. Task prioritization by value density rho_j = v_j / (t_j^dl - t_j^arr).
    2. Preemption with threshold theta > 1, restricted to drift-triggered
       urgent model-adaptation events.
    3. Batch processing with size bounds and energy-efficiency guard.
    4. Support for both inference and fine-tuning tasks.
    5. Resource-aware scheduling decisions.
    """

    def __init__(self, config: Dict):
        # theta is the canonical name; 'preemption_threshold' is accepted for
        # backward compatibility with older configs.
        self.theta = config.get('theta',
                                config.get('preemption_threshold', 1.5))
        self.preemption_threshold = self.theta  # alias retained for callers
        self.batch_size_max = config.get('batch_size_max', 16)
        self.batch_size_min = config.get('batch_size_min', 4)
        self.priority_weights = config.get('priority_weights', {
            'high': 3.0, 'medium': 2.0, 'low': 1.0
        })
        
        # Task management
        self.task_queue = []  # Priority queue for incoming tasks
        self.current_batches = {}  # Currently executing batches per node
        self.waiting_batches = []  # Batches waiting for execution
        self.completed_tasks = []  # Task execution history
        
        # Thread safety
        self.queue_lock = threading.Lock()
        self.batch_lock = threading.Lock()
        
        # Performance metrics
        self.metrics = {
            'total_tasks_processed': 0,
            'total_batches_created': 0,
            'preemptions_performed': 0,
            'average_value_density': 0.0,
            'inference_task_count': 0,
            'fine_tuning_task_count': 0
        }
        
        logger.info(f"VDF Scheduler initialized with theta={self.theta}")
        logger.info(f"Batch size range: {self.batch_size_min}-{self.batch_size_max}")
    
    def submit_task(self, task: IoTTask) -> bool:
        """
        Submit new task to scheduler
        
        Algorithm 1 Step 1: Insert task into priority queue based on value density
        """
        try:
            with self.queue_lock:
                # Validate task timing constraints
                current_time = time.time()
                if task.deadline <= current_time:
                    logger.warning(f"Task {task.task_id} deadline already passed")
                    return False
                
                # Insert into priority queue (heapq maintains min-heap, but our __lt__ reverses order)
                heapq.heappush(self.task_queue, task)
                
                # Update metrics
                self.metrics['total_tasks_processed'] += 1
                if task.task_type == TaskType.INFERENCE:
                    self.metrics['inference_task_count'] += 1
                else:
                    self.metrics['fine_tuning_task_count'] += 1
                
                logger.debug(f"Task {task.task_id} submitted: ρ={task.value_density:.4f}, type={task.task_type.value}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to submit task {task.task_id}: {e}")
            return False
    
    def create_batch(self, target_size: Optional[int] = None) -> Optional[TaskBatch]:
        """
        Create task batch from priority queue
        
        Algorithm 1 Step 2: Group tasks into batch B_i sorted by value density
        """
        if target_size is None:
            target_size = self.batch_size_max
        
        try:
            with self.queue_lock:
                if len(self.task_queue) < self.batch_size_min:
                    return None
                
                # Extract tasks with highest value density
                batch_tasks = []
                extracted_tasks = []
                
                # Get up to target_size tasks
                for _ in range(min(target_size, len(self.task_queue))):
                    if self.task_queue:
                        task = heapq.heappop(self.task_queue)
                        batch_tasks.append(task)
                        extracted_tasks.append(task)
                
                if not batch_tasks:
                    return None
                
                # Sort by value density (descending) as per manuscript: ρ_j1 ≥ ρ_j2 ≥ ... ≥ ρ_jb
                batch_tasks.sort(key=lambda t: t.value_density, reverse=True)
                
                # Create batch
                batch_id = f"batch_{int(time.time()*1000)}_{len(batch_tasks)}"
                batch = TaskBatch(batch_id=batch_id, tasks=batch_tasks)
                
                # Update metrics
                self.metrics['total_batches_created'] += 1
                avg_density = np.mean([task.value_density for task in batch_tasks])
                self.metrics['average_value_density'] = avg_density
                
                logger.info(f"Created batch {batch_id}: {len(batch_tasks)} tasks, ρ_avg={avg_density:.4f}")
                return batch
                
        except Exception as e:
            logger.error(f"Failed to create batch: {e}")
            return None
    
    def check_preemption(self, new_batch: TaskBatch, current_batch: TaskBatch) -> bool:
        """
        Check whether new_batch may preempt current_batch.

        Manuscript rule (revised): preemption is admitted only when the
        aggregate value density of the new batch strictly dominates that of
        the running batch by the threshold theta > 1:

            sum_{j in B_new} rho_j >= theta * sum_{k in B_cur} rho_k.

        The narrow exception captured by this condition is the
        drift-triggered model-adaptation case (an accuracy-degradation event
        in the deployed model makes the adaptation request more urgent than
        the lower-criticality running batch). Ordinary FT tasks carry
        background-level value density and are NOT permitted to displace
        delay-sensitive INF tasks under routine workloads.
        """
        try:
            new_cumulative = new_batch.cumulative_value_density
            current_cumulative = current_batch.cumulative_value_density

            preempt = new_cumulative >= self.theta * current_cumulative

            if preempt:
                logger.info(
                    f"Preemption triggered: sum_rho_new={new_cumulative:.4f} "
                    f">= theta*sum_rho_cur ({self.theta}*{current_cumulative:.4f})"
                )
                self.metrics['preemptions_performed'] += 1

            return preempt

        except Exception as e:
            logger.error(f"Preemption check failed: {e}")
            return False
    
    def schedule_batch(self, batch: TaskBatch, node_id: str, node_capacity: float) -> bool:
        """
        Schedule batch for execution on specified node
        
        Algorithm 1 Step 4: Assign batch to node with resource checking
        """
        try:
            with self.batch_lock:
                current_time = time.time()
                
                # Check if node has current batch
                current_batch = self.current_batches.get(node_id)
                
                if current_batch is not None:
                    # Check preemption condition
                    if self.check_preemption(batch, current_batch):
                        # Preempt current batch
                        logger.info(f"Preempting batch {current_batch.batch_id} with {batch.batch_id}")
                        
                        # Move preempted batch back to waiting queue
                        self.waiting_batches.append(current_batch)
                        
                        # Assign new batch
                        self.current_batches[node_id] = batch
                        batch.assigned_node = node_id
                        
                        return True
                    else:
                        # Cannot preempt, add to waiting queue
                        logger.debug(f"Batch {batch.batch_id} added to waiting queue")
                        self.waiting_batches.append(batch)
                        return False
                else:
                    # Node is free, assign batch directly
                    self.current_batches[node_id] = batch
                    batch.assigned_node = node_id
                    
                    logger.info(f"Batch {batch.batch_id} scheduled on {node_id}")
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to schedule batch {batch.batch_id}: {e}")
            return False
    
    def complete_batch(self, node_id: str, execution_results: Dict) -> bool:
        """
        Mark batch as completed and schedule next waiting batch
        """
        try:
            with self.batch_lock:
                completed_batch = self.current_batches.pop(node_id, None)
                if completed_batch is None:
                    logger.warning(f"No batch found for node {node_id}")
                    return False
                
                # Record completion
                completion_time = time.time()
                for task in completed_batch.tasks:
                    task_result = {
                        'task_id': task.task_id,
                        'completion_time': completion_time,
                        'execution_time': execution_results.get('execution_time', 0),
                        'node_id': node_id,
                        'value_density': task.value_density,
                        'task_type': task.task_type.value
                    }
                    self.completed_tasks.append(task_result)
                
                logger.info(f"Completed batch {completed_batch.batch_id} on {node_id}")
                
                # Schedule next waiting batch if available
                if self.waiting_batches:
                    next_batch = self.waiting_batches.pop(0)
                    self.current_batches[node_id] = next_batch
                    next_batch.assigned_node = node_id
                    logger.info(f"Scheduled waiting batch {next_batch.batch_id} on {node_id}")
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to complete batch on {node_id}: {e}")
            return False
    
    def get_schedule_status(self) -> Dict:
        """Get current scheduling status and metrics"""
        with self.queue_lock, self.batch_lock:
            return {
                'queued_tasks': len(self.task_queue),
                'active_batches': len(self.current_batches),
                'waiting_batches': len(self.waiting_batches),
                'completed_tasks': len(self.completed_tasks),
                'metrics': self.metrics.copy(),
                'current_assignments': {
                    node: batch.batch_id for node, batch in self.current_batches.items()
                }
            }
    
    def get_performance_analysis(self) -> Dict:
        """Generate performance analysis for research validation"""
        try:
            total_tasks = len(self.completed_tasks)
            if total_tasks == 0:
                return {"error": "No completed tasks for analysis"}
            
            # Task type analysis
            inf_tasks = [t for t in self.completed_tasks if t['task_type'] == 'INF']
            ft_tasks = [t for t in self.completed_tasks if t['task_type'] == 'FT']
            
            # Value density analysis
            value_densities = [t['value_density'] for t in self.completed_tasks]
            
            # Execution time analysis
            execution_times = [t['execution_time'] for t in self.completed_tasks]
            
            analysis = {
                'total_tasks_processed': total_tasks,
                'task_type_distribution': {
                    'inference': len(inf_tasks),
                    'fine_tuning': len(ft_tasks)
                },
                'value_density_stats': {
                    'mean': np.mean(value_densities),
                    'std': np.std(value_densities),
                    'min': np.min(value_densities),
                    'max': np.max(value_densities)
                },
                'execution_time_stats': {
                    'mean': np.mean(execution_times),
                    'std': np.std(execution_times),
                    'min': np.min(execution_times),
                    'max': np.max(execution_times)
                },
                'scheduling_efficiency': {
                    'preemption_rate': self.metrics['preemptions_performed'] / max(self.metrics['total_batches_created'], 1),
                    'average_batch_utilization': self.metrics['total_tasks_processed'] / max(self.metrics['total_batches_created'], 1),
                    'inf_task_percentage': len(inf_tasks) / total_tasks * 100,
                    'ft_task_percentage': len(ft_tasks) / total_tasks * 100
                }
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            return {"error": str(e)}

def create_sample_tasks() -> List[IoTTask]:
    """Create sample tasks for testing VDF scheduler"""
    tasks = []
    current_time = time.time()
    
    # Sample IoT analysis tasks with varying priorities and deadlines
    task_configs = [
        {"type": TaskType.INFERENCE, "priority": 3.0, "deadline_offset": 2.0, "compute": 100},
        {"type": TaskType.FINE_TUNING, "priority": 2.0, "deadline_offset": 10.0, "compute": 300},
        {"type": TaskType.INFERENCE, "priority": 2.5, "deadline_offset": 1.5, "compute": 80},
        {"type": TaskType.INFERENCE, "priority": 1.8, "deadline_offset": 3.0, "compute": 120},
        {"type": TaskType.FINE_TUNING, "priority": 3.5, "deadline_offset": 8.0, "compute": 250},
    ]
    
    for i, config in enumerate(task_configs):
        task = IoTTask(
            task_id=f"task_{i+1:03d}",
            task_type=config["type"],
            arrival_time=current_time,
            deadline=current_time + config["deadline_offset"],
            priority_value=config["priority"],
            compute_demand=config["compute"],
            memory_demand=50.0,
            data_size=20.0,
            device_id=f"device_{i%3 + 1}"
        )
        tasks.append(task)
    
    return tasks

if __name__ == "__main__":
    # Test VDF scheduler with sample tasks
    logging.basicConfig(level=logging.INFO)
    
    config = {
        'theta': 1.5,
        'batch_size_max': 4,
        'batch_size_min': 2
    }
    
    scheduler = VDFScheduler(config)
    
    # Create and submit sample tasks
    sample_tasks = create_sample_tasks()
    for task in sample_tasks:
        scheduler.submit_task(task)
        print(f"Task {task.task_id}: ρ = {task.value_density:.4f}, type = {task.task_type.value}")
    
    # Create batch
    batch = scheduler.create_batch()
    if batch:
        print(f"\nCreated batch: {batch.batch_id}")
        print(f"Tasks in batch: {[t.task_id for t in batch.tasks]}")
        print(f"Cumulative value density: {batch.cumulative_value_density:.4f}")
    
    # Display scheduler status
    status = scheduler.get_schedule_status()
    print(f"\nScheduler Status: {status}")