#!/usr/bin/env python3

import math
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import logging

@dataclass
class DeviceSpecs:
    """IoT device specifications for transmission energy calculation"""
    device_type: str
    tx_power_dbm: float  # Transmission power in dBm
    tx_power_watts: float  # Transmission power in watts
    rx_power_watts: float  # Reception power in watts
    idle_power_watts: float  # Idle power consumption
    frequency_ghz: float  # Operating frequency in GHz
    antenna_gain_db: float  # Antenna gain in dB
    battery_capacity_wh: float  # Battery capacity in watt-hours

@dataclass
class TransmissionParameters:
    """Transmission parameters for energy calculation"""
    data_size_bytes: int
    transmission_rate_bps: int
    distance_meters: float
    path_loss_exponent: float = 2.0
    noise_power_dbm: float = -104.0  # Thermal noise floor
    required_snr_db: float = 10.0  # Required SNR for successful transmission

class TransmissionEnergyCalculator:
    """
    Quantifies transmission energy consumption for IoT devices in LL-CIoT framework
    Addresses reviewer concern about lack of empirical transmission energy data
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Define common IoT device specifications
        self.device_specs = {
            'smart_sensor': DeviceSpecs(
                device_type='smart_sensor',
                tx_power_dbm=10,  # 10 dBm = 10 mW
                tx_power_watts=0.01,
                rx_power_watts=0.008,
                idle_power_watts=0.001,
                frequency_ghz=2.4,
                antenna_gain_db=2.0,
                battery_capacity_wh=3.7  # Typical AA battery
            ),
            'smart_home_hub': DeviceSpecs(
                device_type='smart_home_hub',
                tx_power_dbm=20,  # 20 dBm = 100 mW
                tx_power_watts=0.1,
                rx_power_watts=0.08,
                idle_power_watts=0.5,
                frequency_ghz=5.0,
                antenna_gain_db=5.0,
                battery_capacity_wh=50.0
            ),
            'wearable_device': DeviceSpecs(
                device_type='wearable_device', 
                tx_power_dbm=0,  # 0 dBm = 1 mW (low power)
                tx_power_watts=0.001,
                rx_power_watts=0.0008,
                idle_power_watts=0.0001,
                frequency_ghz=2.4,
                antenna_gain_db=0.0,
                battery_capacity_wh=0.5  # Small battery
            )
        }
    
    def calculate_path_loss(self, distance_m: float, frequency_ghz: float, 
                           path_loss_exponent: float = 2.0) -> float:
        """
        Calculate path loss using simplified free space + additional losses model
        
        Args:
            distance_m: Distance in meters
            frequency_ghz: Frequency in GHz
            path_loss_exponent: Path loss exponent (2.0 for free space, higher for obstacles)
            
        Returns:
            Path loss in dB
        """
        # Free space path loss formula: 20*log10(4*pi*d*f/c)
        # where d=distance, f=frequency, c=speed of light
        if distance_m <= 0:
            return 0.0
            
        frequency_hz = frequency_ghz * 1e9
        c = 3e8  # Speed of light in m/s
        
        free_space_loss_db = 20 * math.log10((4 * math.pi * distance_m * frequency_hz) / c)
        
        # Add additional losses for realistic IoT environments
        # Additional losses account for multipath, shadowing, etc.
        additional_loss_db = 10 * (path_loss_exponent - 2.0) * math.log10(distance_m)
        
        total_path_loss_db = free_space_loss_db + additional_loss_db
        
        return total_path_loss_db
    
    def calculate_required_tx_power(self, params: TransmissionParameters, 
                                   device: DeviceSpecs) -> float:
        """
        Calculate required transmission power to achieve target SNR
        
        Args:
            params: Transmission parameters
            device: Device specifications
            
        Returns:
            Required transmission power in watts
        """
        # Calculate path loss
        path_loss_db = self.calculate_path_loss(
            params.distance_meters, 
            device.frequency_ghz,
            params.path_loss_exponent
        )
        
        # Calculate required received power
        # P_rx = Noise_power + SNR_required
        noise_power_watts = 10**((params.noise_power_dbm - 30) / 10)
        required_rx_power_dbm = params.noise_power_dbm + params.required_snr_db
        required_rx_power_watts = 10**((required_rx_power_dbm - 30) / 10)
        
        # Calculate required transmission power
        # P_tx = P_rx + Path_loss - Antenna_gains
        required_tx_power_dbm = (required_rx_power_dbm + path_loss_db - 
                                device.antenna_gain_db)
        required_tx_power_watts = 10**((required_tx_power_dbm - 30) / 10)
        
        # Use device's maximum power if required power exceeds it
        actual_tx_power_watts = min(required_tx_power_watts, device.tx_power_watts)
        
        return actual_tx_power_watts
    
    def calculate_transmission_time(self, data_size_bytes: int, 
                                   transmission_rate_bps: int) -> float:
        """
        Calculate transmission time including protocol overhead
        
        Args:
            data_size_bytes: Data size in bytes
            transmission_rate_bps: Transmission rate in bits per second
            
        Returns:
            Transmission time in seconds
        """
        # Add protocol overhead (headers, ACKs, etc.)
        # Typical overhead: 20% for WiFi, 15% for Bluetooth, 10% for LoRa
        overhead_factor = 1.2  # 20% overhead
        
        total_bits = data_size_bytes * 8 * overhead_factor
        transmission_time_s = total_bits / transmission_rate_bps
        
        return transmission_time_s
    
    def calculate_transmission_energy(self, params: TransmissionParameters,
                                    device_type: str) -> Dict[str, float]:
        """
        Calculate comprehensive transmission energy consumption
        
        Args:
            params: Transmission parameters
            device_type: Type of IoT device
            
        Returns:
            Dictionary with energy breakdown
        """
        if device_type not in self.device_specs:
            raise ValueError(f"Unknown device type: {device_type}")
        
        device = self.device_specs[device_type]
        
        # Calculate transmission time
        tx_time = self.calculate_transmission_time(
            params.data_size_bytes, 
            params.transmission_rate_bps
        )
        
        # Calculate required transmission power for this distance
        required_tx_power = self.calculate_required_tx_power(params, device)
        
        # Energy components calculation
        # 1. Transmission energy (actual RF energy)
        tx_energy_joules = required_tx_power * tx_time
        
        # 2. Circuit energy (power amplifier, digital processing, etc.)
        # Circuit power is typically 2-5x the RF power for efficiency reasons
        circuit_efficiency = 0.3  # 30% efficiency typical for IoT devices
        circuit_power_watts = required_tx_power / circuit_efficiency - required_tx_power
        circuit_energy_joules = circuit_power_watts * tx_time
        
        # 3. Protocol processing energy (MAC layer, error correction, etc.)
        # Estimated as 10% of total transmission energy
        protocol_energy_joules = (tx_energy_joules + circuit_energy_joules) * 0.1
        
        # 4. Reception acknowledgment energy (for reliable protocols)
        ack_time = 0.001  # 1ms typical ACK time
        rx_energy_joules = device.rx_power_watts * ack_time
        
        # Total transmission energy
        total_energy_joules = (tx_energy_joules + circuit_energy_joules + 
                              protocol_energy_joules + rx_energy_joules)
        
        return {
            'tx_energy_joules': tx_energy_joules,
            'circuit_energy_joules': circuit_energy_joules,
            'protocol_energy_joules': protocol_energy_joules,
            'rx_ack_energy_joules': rx_energy_joules,
            'total_transmission_energy_joules': total_energy_joules,
            'transmission_time_seconds': tx_time,
            'tx_power_watts': required_tx_power,
            'path_loss_db': self.calculate_path_loss(params.distance_meters, device.frequency_ghz),
            'energy_per_bit_nanojoules': (total_energy_joules * 1e9) / (params.data_size_bytes * 8),
            'battery_life_hours': device.battery_capacity_wh * 3600 / (total_energy_joules / tx_time) if tx_time > 0 else float('inf')
        }
    
    def calculate_cloud_edge_transmission_energy(self, data_transfer_patterns: List[Dict]) -> Dict[str, Any]:
        """
        Calculate transmission energy for cloud-edge collaborative scenarios
        
        Args:
            data_transfer_patterns: List of data transfer scenarios
            
        Returns:
            Comprehensive transmission energy analysis
        """
        total_energy_breakdown = {
            'device_to_edge_energy_j': 0.0,
            'edge_to_cloud_energy_j': 0.0,
            'cloud_to_edge_energy_j': 0.0,
            'edge_to_device_energy_j': 0.0,
            'total_transmission_energy_j': 0.0
        }
        
        scenario_results = []
        
        for pattern in data_transfer_patterns:
            scenario_result = {}
            
            # Device to Edge transmission (task upload)
            if 'device_to_edge' in pattern:
                d2e = pattern['device_to_edge']
                params = TransmissionParameters(
                    data_size_bytes=d2e['data_size_bytes'],
                    transmission_rate_bps=d2e['rate_bps'],
                    distance_meters=d2e['distance_m'],
                    path_loss_exponent=2.5  # Indoor/urban environment
                )
                
                d2e_energy = self.calculate_transmission_energy(params, d2e['device_type'])
                scenario_result['device_to_edge'] = d2e_energy
                total_energy_breakdown['device_to_edge_energy_j'] += d2e_energy['total_transmission_energy_joules']
            
            # Edge to Cloud transmission (complex task offloading)
            if 'edge_to_cloud' in pattern:
                e2c = pattern['edge_to_cloud']
                params = TransmissionParameters(
                    data_size_bytes=e2c['data_size_bytes'],
                    transmission_rate_bps=e2c['rate_bps'],
                    distance_meters=e2c['distance_m'],
                    path_loss_exponent=2.0  # Line-of-sight or fiber
                )
                
                e2c_energy = self.calculate_transmission_energy(params, 'smart_home_hub')
                scenario_result['edge_to_cloud'] = e2c_energy
                total_energy_breakdown['edge_to_cloud_energy_j'] += e2c_energy['total_transmission_energy_joules']
            
            # Cloud to Edge transmission (model updates, results)
            if 'cloud_to_edge' in pattern:
                c2e = pattern['cloud_to_edge']
                params = TransmissionParameters(
                    data_size_bytes=c2e['data_size_bytes'],
                    transmission_rate_bps=c2e['rate_bps'],
                    distance_meters=c2e['distance_m'],
                    path_loss_exponent=2.0
                )
                
                c2e_energy = self.calculate_transmission_energy(params, 'smart_home_hub')
                scenario_result['cloud_to_edge'] = c2e_energy
                total_energy_breakdown['cloud_to_edge_energy_j'] += c2e_energy['total_transmission_energy_joules']
            
            # Edge to Device transmission (results delivery)
            if 'edge_to_device' in pattern:
                e2d = pattern['edge_to_device']
                params = TransmissionParameters(
                    data_size_bytes=e2d['data_size_bytes'],
                    transmission_rate_bps=e2d['rate_bps'],
                    distance_meters=e2d['distance_m'],
                    path_loss_exponent=2.5
                )
                
                e2d_energy = self.calculate_transmission_energy(params, 'smart_home_hub')
                scenario_result['edge_to_device'] = e2d_energy
                total_energy_breakdown['edge_to_device_energy_j'] += e2d_energy['total_transmission_energy_joules']
            
            scenario_results.append(scenario_result)
        
        # Calculate total transmission energy
        total_energy_breakdown['total_transmission_energy_j'] = sum([
            total_energy_breakdown['device_to_edge_energy_j'],
            total_energy_breakdown['edge_to_cloud_energy_j'],
            total_energy_breakdown['cloud_to_edge_energy_j'],
            total_energy_breakdown['edge_to_device_energy_j']
        ])
        
        return {
            'total_energy_breakdown': total_energy_breakdown,
            'scenario_results': scenario_results,
            'average_energy_per_transaction_j': total_energy_breakdown['total_transmission_energy_j'] / len(data_transfer_patterns) if data_transfer_patterns else 0,
            'energy_distribution': {
                'device_edge_percentage': (total_energy_breakdown['device_to_edge_energy_j'] + total_energy_breakdown['edge_to_device_energy_j']) / total_energy_breakdown['total_transmission_energy_j'] * 100 if total_energy_breakdown['total_transmission_energy_j'] > 0 else 0,
                'edge_cloud_percentage': (total_energy_breakdown['edge_to_cloud_energy_j'] + total_energy_breakdown['cloud_to_edge_energy_j']) / total_energy_breakdown['total_transmission_energy_j'] * 100 if total_energy_breakdown['total_transmission_energy_j'] > 0 else 0
            }
        }
    
    def generate_realistic_transmission_scenarios(self) -> List[Dict]:
        """
        Generate realistic transmission scenarios for LL-CIoT evaluation
        
        Returns:
            List of realistic data transfer scenarios
        """
        scenarios = []
        
        # Scenario 1: Smart sensor periodic data upload
        scenarios.append({
            'scenario_name': 'sensor_data_upload',
            'device_to_edge': {
                'device_type': 'smart_sensor',
                'data_size_bytes': 512,  # Sensor data packet
                'rate_bps': 250000,  # 250 kbps
                'distance_m': 20.0   # 20 meters to edge server
            },
            'edge_to_cloud': {
                'data_size_bytes': 1024,  # Aggregated + metadata
                'rate_bps': 10000000,  # 10 Mbps backbone
                'distance_m': 1000.0   # 1km to cloud (represents network hops)
            }
        })
        
        # Scenario 2: Smart home hub task offloading
        scenarios.append({
            'scenario_name': 'hub_task_offloading',
            'device_to_edge': {
                'device_type': 'smart_home_hub',
                'data_size_bytes': 4096,  # Complex task data
                'rate_bps': 1000000,  # 1 Mbps WiFi
                'distance_m': 15.0
            },
            'edge_to_cloud': {
                'data_size_bytes': 8192,  # Task + intermediate results
                'rate_bps': 50000000,  # 50 Mbps
                'distance_m': 1000.0
            },
            'cloud_to_edge': {
                'data_size_bytes': 2048,  # Model updates/results
                'rate_bps': 50000000,
                'distance_m': 1000.0
            },
            'edge_to_device': {
                'data_size_bytes': 1024,  # Final results
                'rate_bps': 1000000,
                'distance_m': 15.0
            }
        })
        
        # Scenario 3: Wearable health monitoring
        scenarios.append({
            'scenario_name': 'wearable_health_monitoring',
            'device_to_edge': {
                'device_type': 'wearable_device',
                'data_size_bytes': 256,  # Health sensor data
                'rate_bps': 125000,  # 125 kbps Bluetooth LE
                'distance_m': 5.0
            },
            'edge_to_cloud': {
                'data_size_bytes': 512,  # Health analysis request
                'rate_bps': 10000000,
                'distance_m': 1000.0
            }
        })
        
        return scenarios

def main():
    """Example usage and validation of transmission energy calculations"""
    logging.basicConfig(level=logging.INFO)
    calculator = TransmissionEnergyCalculator()
    
    print("=== LL-CIoT Transmission Energy Analysis ===")
    print("Quantifying transmission energy consumption for reviewer validation\n")
    
    # Generate realistic scenarios
    scenarios = calculator.generate_realistic_transmission_scenarios()
    
    # Calculate transmission energy for all scenarios
    results = calculator.calculate_cloud_edge_transmission_energy(scenarios)
    
    # Display results
    print("📊 TOTAL ENERGY BREAKDOWN:")
    total_breakdown = results['total_energy_breakdown']
    for component, energy in total_breakdown.items():
        print(f"   {component}: {energy:.6f} J")
    
    print(f"\n📈 ENERGY DISTRIBUTION:")
    distribution = results['energy_distribution']
    print(f"   Device-Edge Communication: {distribution['device_edge_percentage']:.1f}%")
    print(f"   Edge-Cloud Communication: {distribution['edge_cloud_percentage']:.1f}%")
    
    print(f"\n📋 AVERAGE ENERGY PER TRANSACTION: {results['average_energy_per_transaction_j']:.6f} J")
    
    print("\n🔍 INDIVIDUAL SCENARIO ANALYSIS:")
    for i, scenario_result in enumerate(results['scenario_results']):
        scenario_name = scenarios[i]['scenario_name']
        print(f"\n   Scenario: {scenario_name}")
        for transfer_type, energy_data in scenario_result.items():
            total_energy = energy_data['total_transmission_energy_joules']
            energy_per_bit = energy_data['energy_per_bit_nanojoules']
            print(f"      {transfer_type}: {total_energy:.6f} J ({energy_per_bit:.2f} nJ/bit)")

if __name__ == "__main__":
    main()