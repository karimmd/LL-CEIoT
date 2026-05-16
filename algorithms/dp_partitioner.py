#!/usr/bin/env python3
"""
LL-CEIoT Reference Implementation - DP Partitioner (Algorithm 2, revised)

Quantization-aware layer-node mapping with aggregate resource-state tracking
and Pareto pruning, matching the revised Algorithm 2 in the manuscript.

The earlier scalar DP (dp[layer][node] = best delay) checked feasibility
only at the per-layer level. The revised formulation tracks aggregate
node-level memory and compute consumption across all layers assigned to
each node, directly enforcing constraints (C1) and (C2) of P1 in their
accumulated form.

State at (l, n) is a label set S_{l,n}; each label lambda = (cost, R, parent)
records:
    cost   - cumulative placement-stage delay through layer l
    R      - residual resource profile across all nodes
    parent - predecessor-label pointer for backtracking

Transitions are admitted only when the residual on the target node still
covers the quantized layer demand. Pareto pruning then discards labels
that are dominated in both cost and every residual dimension.

Symbols (manuscript Section "CE Collaboration with Lightweight LLM Placement"):
    R^M_n   : residual memory on node n
    R^phi_n : residual compute budget on n  ( phi_n(t) * Delta t )
    Omega_i : batch workload from Stage 1   ( sum_{j in B_i}(tau_prompt + tau_gen) )
    c_l^quant, m_l^quant : quantized layer demand  ( alpha_l * c_l, alpha_l * m_l )
    beta_{l-1,l}         : inter-layer activation size

Complexity: O(|L| * |N|^2 * S), where S is the maximum number of
non-dominated labels retained per (layer, node) state after Pareto pruning.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import time

logger = logging.getLogger(__name__)


class NodeTier(Enum):
    """Node tier classification."""
    CLOUD = "cloud"
    EDGE = "edge"


@dataclass
class ModelLayer:
    """
    Layer descriptor.

    Symbols:
        l           : layer index
        c_l, m_l    : full-precision compute / memory demand
        alpha_l     : quantization factor
        c_l^quant   : alpha_l * c_l
        m_l^quant   : alpha_l * m_l
    """
    layer_id: int
    layer_name: str
    compute_demand: float
    memory_requirement: float
    quantization_factor: float = 1.0

    def __post_init__(self):
        if self.quantization_factor <= 0 or self.quantization_factor > 1:
            raise ValueError(f"Invalid quantization factor: {self.quantization_factor}")

    @property
    def quantized_compute_demand(self) -> float:
        return self.quantization_factor * self.compute_demand

    @property
    def quantized_memory_requirement(self) -> float:
        return self.quantization_factor * self.memory_requirement


@dataclass
class NodeResource:
    """
    Node resource descriptor.

    Symbols:
        n              : node index
        phi_n(t)       : compute capacity at time t
        M_n            : memory capacity
        b_{n,m}        : inter-node bandwidth
    """
    node_id: str
    tier: NodeTier
    gpu_id: int
    compute_capacity: float
    memory_capacity: float
    bandwidth_to_nodes: Dict[str, float]


@dataclass
class LayerAssignment:
    """
    Solution element X_{l,n} = 1 with delay decomposition.
    """
    layer_id: int
    node_id: str
    processing_delay: float
    communication_delay: float
    total_delay: float
    quantization_factor: float


@dataclass
class ResourceLabel:
    """
    A non-dominated label in S_{l,n}.

    Fields:
        cost     : cumulative placement-stage delay through this layer
        residual : per-node residual {node_id: {'mem': R^M_n, 'compute': R^phi_n}}
        parent   : pointer to the predecessor label (for backtracking)
        node_id  : node carrying the current layer (l)
        layer_id : current layer index l
    """
    cost: float
    residual: Dict[str, Dict[str, float]]
    parent: Optional["ResourceLabel"]
    node_id: str
    layer_id: int

    def dominates(self, other: "ResourceLabel") -> bool:
        """True iff self weakly dominates other in cost and every residual axis."""
        if self.cost > other.cost:
            return False
        for n_id, res in self.residual.items():
            other_res = other.residual[n_id]
            if res['mem'] < other_res['mem'] or res['compute'] < other_res['compute']:
                return False
        # weak domination requires at least one strict inequality
        if self.cost < other.cost:
            return True
        for n_id, res in self.residual.items():
            other_res = other.residual[n_id]
            if res['mem'] > other_res['mem'] or res['compute'] > other_res['compute']:
                return True
        return False


def _copy_residual(residual: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    return {n: {'mem': r['mem'], 'compute': r['compute']} for n, r in residual.items()}


class DPPartitioner:
    """
    Algorithm 2 (revised): quantization-aware layer-node mapping with
    aggregate resource-state tracking and Pareto pruning.
    """

    def __init__(self, config: Dict):
        self.communication_penalty = config.get('communication_penalty', 0.0)
        self.memory_penalty = config.get('memory_penalty', 0.0)
        self.timeslot_duration_ms = config.get('timeslot_duration_ms', 100.0)
        self.delta_t_seconds = config.get('delta_t_seconds', self.timeslot_duration_ms / 1000.0)
        self.omega_default = config.get('omega_default', 1.0)
        self.max_optimization_time_ms = config.get('max_optimization_time_ms', 50.0)
        self.resource_update_interval_ms = config.get('resource_update_interval_ms', 10.0)
        self.max_labels_per_state = config.get('max_labels_per_state', 64)

        # Replaces the legacy scalar dp_table with a label set.
        # label_sets[(layer_idx, node_id)] = List[ResourceLabel]
        self.label_sets: Dict[Tuple[int, str], List[ResourceLabel]] = {}

        self.inter_layer_data_sizes: Dict[Tuple[int, int], float] = {}
        self.current_timeslot = 0
        self.last_resource_update = 0.0
        self.cached_solutions: Dict[str, Tuple[List[LayerAssignment], float]] = {}

        self.optimization_stats = {
            'total_partitioning_time': 0.0,
            'layers_processed': 0,
            'nodes_considered': 0,
            'optimal_assignments': 0,
            'timeslots_processed': 0,
            'real_time_violations': 0,
            'cache_hits': 0,
            'pareto_prunes': 0,
            'max_labels_observed': 0,
            'average_optimization_time_ms': 0.0,
            'complexity_validation': {
                'max_layers_processed': 0,
                'max_nodes_processed': 0,
                'theoretical_complexity': 'O(|L| * |N|^2 * S)',
                'measured_label_factor': 0.0,
            }
        }

        logger.info("DP Partitioner (revised Algorithm 2) initialized")
        logger.info(f"  delta_t = {self.delta_t_seconds}s, omega_default = {self.omega_default}")
        logger.info(f"  max_labels_per_state = {self.max_labels_per_state}")

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    def set_inter_layer_data_sizes(self, data_sizes: Dict[Tuple[int, int], float]):
        """Set beta_{l,l+1} for every consecutive layer pair."""
        self.inter_layer_data_sizes = data_sizes
        logger.info(f"Set inter-layer data sizes for {len(data_sizes)} layer pairs")

    # ------------------------------------------------------------------
    # Algorithm 2 (revised) - core DP with label sets
    # ------------------------------------------------------------------

    def _processing_delay(self, layer: ModelLayer, node: NodeResource, omega_i: float) -> float:
        """
        Manuscript form:
            d_proc(l, n) = Omega_i * c_l^quant / phi_n(t).
        """
        if node.compute_capacity <= 0:
            return float('inf')
        return omega_i * layer.quantized_compute_demand / node.compute_capacity

    def _communication_delay(self,
                             prev_node: Optional[str],
                             curr_node: str,
                             layer_id: int,
                             nodes: Dict[str, NodeResource]) -> float:
        """
        Manuscript form:
            d_comm = (beta_{l-1,l} / b_{n',n}) * I[n' != n].
        """
        if prev_node is None or prev_node == curr_node:
            return 0.0

        size = self.inter_layer_data_sizes.get((layer_id - 1, layer_id))
        if size is None:
            logger.warning(f"No beta specified for ({layer_id-1},{layer_id}); using 100.0 default")
            size = 100.0

        bw_map = nodes[prev_node].bandwidth_to_nodes
        if curr_node not in bw_map or bw_map[curr_node] <= 0:
            return float('inf')
        return size / bw_map[curr_node]

    def _initial_residual(self, nodes: Dict[str, NodeResource]) -> Dict[str, Dict[str, float]]:
        """
        Initialize R^M_n <- M_n and R^phi_n <- phi_n(t) * Delta t for all n.
        """
        return {
            n_id: {
                'mem': float(node.memory_capacity),
                'compute': float(node.compute_capacity) * float(self.delta_t_seconds),
            }
            for n_id, node in nodes.items()
        }

    def _admissible(self,
                    residual: Dict[str, Dict[str, float]],
                    node_id: str,
                    layer: ModelLayer,
                    omega_i: float) -> bool:
        """
        Admission test:
            R[R^M_n]   >= m_l^quant
            R[R^phi_n] >= Omega_i * c_l^quant
        """
        r = residual[node_id]
        if r['mem'] < layer.quantized_memory_requirement:
            return False
        if r['compute'] < omega_i * layer.quantized_compute_demand:
            return False
        return True

    def _apply_transition(self,
                          residual: Dict[str, Dict[str, float]],
                          node_id: str,
                          layer: ModelLayer,
                          omega_i: float) -> Dict[str, Dict[str, float]]:
        """
        R'[R^M_n]   <- R[R^M_n]   - m_l^quant
        R'[R^phi_n] <- R[R^phi_n] - Omega_i * c_l^quant
        """
        new_residual = _copy_residual(residual)
        new_residual[node_id]['mem'] -= layer.quantized_memory_requirement
        new_residual[node_id]['compute'] -= omega_i * layer.quantized_compute_demand
        return new_residual

    def _pareto_insert(self,
                       label_set: List[ResourceLabel],
                       new_label: ResourceLabel) -> List[ResourceLabel]:
        """
        Insert new_label into label_set under Pareto domination.
        Discards label_a if there exists label_b != label_a with
        cost(b) <= cost(a) and R_b[k] >= R_a[k] for all k.
        """
        for existing in label_set:
            if existing.dominates(new_label):
                self.optimization_stats['pareto_prunes'] += 1
                return label_set

        # new_label may dominate some existing ones
        survivors = [lab for lab in label_set if not new_label.dominates(lab)]
        survivors.append(new_label)

        if len(survivors) > self.max_labels_per_state:
            survivors.sort(key=lambda l: l.cost)
            survivors = survivors[:self.max_labels_per_state]

        if len(survivors) > self.optimization_stats['max_labels_observed']:
            self.optimization_stats['max_labels_observed'] = len(survivors)

        return survivors

    def _initialize_label_sets(self,
                               layers: List[ModelLayer],
                               nodes: Dict[str, NodeResource],
                               omega_i: float):
        """
        Base case: place layer 0 on each feasible node and seed S_{0,n}.
        """
        self.label_sets.clear()
        first_layer = layers[0]
        base_residual = self._initial_residual(nodes)

        for n_id, node in nodes.items():
            if not self._admissible(base_residual, n_id, first_layer, omega_i):
                self.label_sets[(0, n_id)] = []
                continue

            new_residual = self._apply_transition(base_residual, n_id, first_layer, omega_i)
            cost = self._processing_delay(first_layer, node, omega_i)
            label = ResourceLabel(
                cost=cost,
                residual=new_residual,
                parent=None,
                node_id=n_id,
                layer_id=0,
            )
            self.label_sets[(0, n_id)] = [label]

    def _extend_label_sets(self,
                           layer_idx: int,
                           layer: ModelLayer,
                           nodes: Dict[str, NodeResource],
                           omega_i: float):
        """
        DP transition for layer_idx >= 1:
            For each (l-1, n') label, for each candidate n,
            admit if residual on n covers (m_l^quant, Omega_i * c_l^quant);
            new cost = parent cost + proc(n) + comm(n', n);
            update residual on n; insert with Pareto pruning into S_{l, n}.
        """
        for curr_id, curr_node in nodes.items():
            new_set: List[ResourceLabel] = []
            proc = self._processing_delay(layer, curr_node, omega_i)

            for prev_id in nodes.keys():
                prev_set = self.label_sets.get((layer_idx - 1, prev_id), [])
                if not prev_set:
                    continue
                comm = self._communication_delay(prev_id, curr_id, layer_idx, nodes)
                if comm == float('inf'):
                    continue

                for parent_label in prev_set:
                    if not self._admissible(parent_label.residual, curr_id, layer, omega_i):
                        continue

                    new_residual = self._apply_transition(
                        parent_label.residual, curr_id, layer, omega_i)

                    extra = 0.0
                    if curr_node.tier == NodeTier.EDGE and layer.quantization_factor < 0.5:
                        extra += self.memory_penalty * layer.quantized_memory_requirement
                    if prev_id != curr_id and nodes[prev_id].tier != curr_node.tier:
                        extra += self.communication_penalty * comm

                    new_cost = parent_label.cost + proc + comm + extra
                    candidate = ResourceLabel(
                        cost=new_cost,
                        residual=new_residual,
                        parent=parent_label,
                        node_id=curr_id,
                        layer_id=layer_idx,
                    )
                    new_set = self._pareto_insert(new_set, candidate)

            self.label_sets[(layer_idx, curr_id)] = new_set

    def _reconstruct_solution(self,
                              layers: List[ModelLayer],
                              nodes: Dict[str, NodeResource],
                              omega_i: float) -> List[LayerAssignment]:
        """
        Pick (n*, lambda*) = argmin over S_{L-1, n} and walk parent chain.
        """
        last = len(layers) - 1
        best: Optional[ResourceLabel] = None
        for n_id in nodes.keys():
            for label in self.label_sets.get((last, n_id), []):
                if best is None or label.cost < best.cost:
                    best = label
        if best is None:
            raise ValueError("No feasible placement found under aggregate resource constraints")

        logger.info(f"Optimal placement-stage cost: {best.cost:.4f}")

        chain: List[ResourceLabel] = []
        cursor: Optional[ResourceLabel] = best
        while cursor is not None:
            chain.append(cursor)
            cursor = cursor.parent
        chain.reverse()  # now in layer order 0..L-1

        assignments: List[LayerAssignment] = []
        for idx, label in enumerate(chain):
            layer = layers[idx]
            node = nodes[label.node_id]
            proc = self._processing_delay(layer, node, omega_i)
            if idx == 0:
                comm = 0.0
            else:
                comm = self._communication_delay(
                    chain[idx - 1].node_id, label.node_id, idx, nodes)
            assignments.append(LayerAssignment(
                layer_id=idx,
                node_id=label.node_id,
                processing_delay=proc,
                communication_delay=comm,
                total_delay=proc + comm,
                quantization_factor=layer.quantization_factor,
            ))
        return assignments

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def partition_model(self,
                        layers: List[ModelLayer],
                        nodes: Dict[str, NodeResource],
                        omega_i: Optional[float] = None,
                        delta_t_seconds: Optional[float] = None
                        ) -> Tuple[List[LayerAssignment], Dict]:
        """
        Run revised Algorithm 2 with default real-time bounds.

        Args:
            layers           : ordered list of LLM layers (with alpha_l set).
            nodes            : node_id -> NodeResource.
            omega_i          : batch workload Omega_i from Stage 1.
                               Defaults to self.omega_default.
            delta_t_seconds  : timeslot duration. Overrides instance default.
        """
        current_time_ms = time.time() * 1000.0
        deadline_ms = current_time_ms + self.max_optimization_time_ms
        return self._partition_with_time_bounds(
            layers, nodes,
            omega_i if omega_i is not None else self.omega_default,
            delta_t_seconds if delta_t_seconds is not None else self.delta_t_seconds,
            current_time_ms, deadline_ms,
        )

    def partition_model_with_timeslot(self,
                                      layers: List[ModelLayer],
                                      nodes: Dict[str, NodeResource],
                                      current_time_ms: Optional[float] = None,
                                      deadline_ms: Optional[float] = None,
                                      omega_i: Optional[float] = None,
                                      delta_t_seconds: Optional[float] = None
                                      ) -> Tuple[List[LayerAssignment], Dict]:
        if current_time_ms is None:
            current_time_ms = time.time() * 1000.0
        if deadline_ms is None:
            deadline_ms = current_time_ms + self.max_optimization_time_ms
        return self._partition_with_time_bounds(
            layers, nodes,
            omega_i if omega_i is not None else self.omega_default,
            delta_t_seconds if delta_t_seconds is not None else self.delta_t_seconds,
            current_time_ms, deadline_ms,
        )

    def _partition_with_time_bounds(self,
                                    layers: List[ModelLayer],
                                    nodes: Dict[str, NodeResource],
                                    omega_i: float,
                                    delta_t_seconds: float,
                                    start_time_ms: float,
                                    deadline_ms: float
                                    ) -> Tuple[List[LayerAssignment], Dict]:
        if not layers:
            raise ValueError("No layers provided")
        if not nodes:
            raise ValueError("No nodes provided")

        # delta_t flows through residual init, so re-pin instance value for this run
        self.delta_t_seconds = delta_t_seconds

        partition_start = time.time()
        wall_ms = time.time() * 1000.0
        self.current_timeslot = int(start_time_ms / max(self.timeslot_duration_ms, 1e-9))

        # Cache lookup
        cache_key = self._generate_cache_key(layers, nodes, omega_i)
        if cache_key in self.cached_solutions:
            cached_solution, cached_time = self.cached_solutions[cache_key]
            if (wall_ms - cached_time) < (self.timeslot_duration_ms * 2):
                self.optimization_stats['cache_hits'] += 1
                return cached_solution, {'optimization_method': 'cached',
                                         'cache_age_ms': wall_ms - cached_time}

        # Complexity tracking
        cv = self.optimization_stats['complexity_validation']
        cv['max_layers_processed'] = max(cv['max_layers_processed'], len(layers))
        cv['max_nodes_processed'] = max(cv['max_nodes_processed'], len(nodes))

        try:
            self._initialize_label_sets(layers, nodes, omega_i)

            for l_idx in range(1, len(layers)):
                if (time.time() * 1000.0) > deadline_ms:
                    logger.info(f"Time bound reached at layer {l_idx}; returning partial solution")
                    self.optimization_stats['real_time_violations'] += 1
                    return self._partial_solution(layers, l_idx, nodes, omega_i), \
                           {'method': 'partial_time_bounded', 'layers_completed': l_idx}
                self._extend_label_sets(l_idx, layers[l_idx], nodes, omega_i)

            assignments = self._reconstruct_solution(layers, nodes, omega_i)

            partition_time = time.time() - partition_start
            stats = self._calculate_partitioning_stats(layers, assignments, nodes, partition_time)
            stats['real_time_metrics'] = {
                'timeslot': self.current_timeslot,
                'optimization_time_ms': partition_time * 1000.0,
                'deadline_met': partition_time * 1000.0 <= self.max_optimization_time_ms,
                'omega_i': omega_i,
                'delta_t_seconds': delta_t_seconds,
                'max_labels_per_state_observed': self.optimization_stats['max_labels_observed'],
            }

            self.cached_solutions[cache_key] = (assignments, wall_ms)
            if len(self.cached_solutions) > 16:
                oldest = min(self.cached_solutions.keys(),
                             key=lambda k: self.cached_solutions[k][1])
                del self.cached_solutions[oldest]

            self.optimization_stats['total_partitioning_time'] += partition_time
            self.optimization_stats['layers_processed'] += len(layers)
            self.optimization_stats['nodes_considered'] += len(nodes)
            self.optimization_stats['optimal_assignments'] += len(assignments)
            self.optimization_stats['timeslots_processed'] += 1

            n = self.optimization_stats['timeslots_processed']
            avg_old = self.optimization_stats['average_optimization_time_ms']
            self.optimization_stats['average_optimization_time_ms'] = (
                (avg_old * (n - 1) + partition_time * 1000.0) / n
            )

            return assignments, stats

        except Exception as exc:
            logger.error(f"Revised DP partitioning failed: {exc}")
            return self._fallback_solution(layers, nodes, omega_i), \
                   {'method': 'fallback_error', 'error': str(exc)}

    # ------------------------------------------------------------------
    # Fallback / partial / cache key
    # ------------------------------------------------------------------

    def _partial_solution(self,
                          layers: List[ModelLayer],
                          completed_layers: int,
                          nodes: Dict[str, NodeResource],
                          omega_i: float) -> List[LayerAssignment]:
        if completed_layers <= 0:
            return self._fallback_solution(layers, nodes, omega_i)
        partial_layers = layers[:completed_layers]
        return self._reconstruct_solution(partial_layers, nodes, omega_i)

    def _fallback_solution(self,
                           layers: List[ModelLayer],
                           nodes: Dict[str, NodeResource],
                           omega_i: float) -> List[LayerAssignment]:
        """
        Greedy fallback: place each layer on the node with smallest
        per-layer processing delay regardless of cumulative residual.
        """
        assignments = []
        for layer in layers:
            best = min(
                nodes.items(),
                key=lambda kv: self._processing_delay(layer, kv[1], omega_i),
            )[0]
            proc = self._processing_delay(layer, nodes[best], omega_i)
            assignments.append(LayerAssignment(
                layer_id=layer.layer_id,
                node_id=best,
                processing_delay=proc,
                communication_delay=0.0,
                total_delay=proc,
                quantization_factor=layer.quantization_factor,
            ))
        return assignments

    def _generate_cache_key(self,
                            layers: List[ModelLayer],
                            nodes: Dict[str, NodeResource],
                            omega_i: float) -> str:
        layer_sig = hash(tuple(
            (l.layer_id, l.compute_demand, l.memory_requirement, l.quantization_factor)
            for l in layers))
        node_sig = hash(tuple(
            (n, node.compute_capacity, node.memory_capacity)
            for n, node in nodes.items()))
        return f"{layer_sig}_{node_sig}_{omega_i}_{self.delta_t_seconds}_{self.current_timeslot}"

    # ------------------------------------------------------------------
    # Stats / export
    # ------------------------------------------------------------------

    def _calculate_partitioning_stats(self,
                                      layers: List[ModelLayer],
                                      assignments: List[LayerAssignment],
                                      nodes: Dict[str, NodeResource],
                                      partition_time: float) -> Dict:
        total_proc = sum(a.processing_delay for a in assignments)
        total_comm = sum(a.communication_delay for a in assignments)
        total_delay = total_proc + total_comm

        cloud_assignments = sum(1 for a in assignments if nodes[a.node_id].tier == NodeTier.CLOUD)
        edge_assignments = len(assignments) - cloud_assignments

        inter_tier = 0
        intra_tier = 0
        for i in range(1, len(assignments)):
            if assignments[i].communication_delay > 0:
                if nodes[assignments[i - 1].node_id].tier != nodes[assignments[i].node_id].tier:
                    inter_tier += 1
                else:
                    intra_tier += 1

        q_levels = [a.quantization_factor for a in assignments]
        avg_q = float(np.mean(q_levels)) if q_levels else 0.0

        node_util: Dict[str, Dict[str, float]] = {}
        for node_id, node in nodes.items():
            assigned = [a for a in assignments if a.node_id == node_id]
            if not assigned:
                continue
            comp_used = sum(layers[a.layer_id].quantized_compute_demand for a in assigned)
            mem_used = sum(layers[a.layer_id].quantized_memory_requirement for a in assigned)
            node_util[node_id] = {
                'compute_utilization': comp_used / max(node.compute_capacity, 1e-9),
                'memory_utilization': mem_used / max(node.memory_capacity, 1e-9),
                'layers_assigned': len(assigned),
            }

        return {
            'total_delay': total_delay,
            'processing_delay': total_proc,
            'communication_delay': total_comm,
            'partition_time': partition_time,
            'cloud_assignments': cloud_assignments,
            'edge_assignments': edge_assignments,
            'total_layers': len(assignments),
            'inter_tier_communications': inter_tier,
            'intra_tier_communications': intra_tier,
            'average_quantization_factor': avg_q,
            'node_utilization': node_util,
            'label_set_size': sum(len(v) for v in self.label_sets.values()),
            'max_labels_per_state': self.optimization_stats['max_labels_observed'],
            'pareto_prunes': self.optimization_stats['pareto_prunes'],
            'backtrack_steps': len(assignments),
        }

    def export_assignments(self, assignments: List[LayerAssignment], filename: str):
        export_data = {
            'timestamp': time.time(),
            'algorithm': 'DP_Partitioner_Algorithm_2_revised',
            'assignments': [
                {
                    'layer_id': a.layer_id,
                    'node_id': a.node_id,
                    'processing_delay': a.processing_delay,
                    'communication_delay': a.communication_delay,
                    'total_delay': a.total_delay,
                    'quantization_factor': a.quantization_factor,
                }
                for a in assignments
            ],
            'summary': {
                'total_layers': len(assignments),
                'total_delay': sum(a.total_delay for a in assignments),
                'total_processing_delay': sum(a.processing_delay for a in assignments),
                'total_communication_delay': sum(a.communication_delay for a in assignments),
                'optimization_stats': self.optimization_stats,
            },
        }
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        logger.info(f"Assignments exported to {filename}")


# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Demo: units chosen so that one full timeslot can absorb the batch.
    # phi_n is in GFLOPs, c_l in GFLOPs/token, Omega_i in tokens, Delta t in seconds.
    config = {
        'communication_penalty': 0.0,
        'memory_penalty': 0.0,
        'delta_t_seconds': 1.0,
        'omega_default': 100.0,  # cumulative tokens in the Stage-1 batch
    }
    partitioner = DPPartitioner(config)

    layers = [
        ModelLayer(0, "embedding",   1.0, 500.0, 1.0),
        ModelLayer(1, "attention_0", 2.0, 800.0, 0.5),
        ModelLayer(2, "mlp_0",       1.5, 600.0, 0.5),
        ModelLayer(3, "attention_1", 2.0, 800.0, 0.25),
        ModelLayer(4, "output",      0.8, 400.0, 1.0),
    ]
    nodes = {
        'cloud_server': NodeResource('cloud_server', NodeTier.CLOUD, 0,
                                     800.0, 11000.0, {'edge_server': 50.0}),
        'edge_server':  NodeResource('edge_server',  NodeTier.EDGE,  0,
                                     200.0,  4000.0, {'cloud_server': 50.0}),
    }
    partitioner.set_inter_layer_data_sizes({(0, 1): 50.0, (1, 2): 40.0,
                                            (2, 3): 40.0, (3, 4): 30.0})

    assignments, stats = partitioner.partition_model(layers, nodes, omega_i=100.0)

    print("DP Partitioner Results (Algorithm 2, revised):")
    print("=" * 60)
    for a in assignments:
        print(f"  layer {a.layer_id} -> {a.node_id}  "
              f"(proc {a.processing_delay:.3f}, comm {a.communication_delay:.3f}, "
              f"alpha {a.quantization_factor})")
    print(f"\nstats: {json.dumps(stats, indent=2, default=str)}")

    partitioner.export_assignments(assignments, "dp_partitioner_results.json")
