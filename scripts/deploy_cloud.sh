#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Check if running as root or with sudo
if [[ $EUID -eq 0 ]]; then
    echo "Warning: Running as root. Consider using a non-root user for deployment."
fi

# Function to check dependencies
check_dependencies() {
    echo "Checking dependencies..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo "ERROR: Python 3 is required but not installed"
        exit 1
    fi
    
    # Check pip
    if ! command -v pip3 &> /dev/null; then
        echo "ERROR: pip3 is required but not installed"
        exit 1
    fi
    
    # Check required Python packages
    python3 -c "import torch, transformers, pandas, numpy" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "ERROR: Missing required Python packages"
        echo "Install with: pip3 install torch transformers pandas numpy"
        exit 1
    fi
    
    echo "Dependencies check passed"
}

# Function to check hardware
check_hardware() {
    echo "Checking hardware configuration..."
    
    # Check GPU
    if command -v nvidia-smi &> /dev/null; then
        echo "CUDA GPU detected:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
        
        # Verify minimum memory (need ~8GB for large models)
        GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        if [ "$GPU_MEM" -lt 8000 ]; then
            echo "WARNING: GPU memory ($GPU_MEM MB) may be insufficient for large models"
        fi
    else
        echo "WARNING: nvidia-smi not found. GPU inference may not be available."
    fi
    
    # Check system memory
    TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
    echo "System memory: ${TOTAL_MEM}GB"
    if [ "$TOTAL_MEM" -lt 16 ]; then
        echo "WARNING: System memory (${TOTAL_MEM}GB) may be insufficient"
    fi
    
    # Check CPU cores
    CPU_CORES=$(nproc)
    echo "CPU cores: $CPU_CORES"
}

# Function to setup virtual environment
setup_environment() {
    echo "Setting up Python environment..."
    
    cd "$PROJECT_ROOT"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        echo "Created virtual environment"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    else
        # Install basic requirements
        pip install torch transformers pandas numpy psutil asyncio-throttle
    fi
    
    echo "Environment setup complete"
}

# Function to check model files
check_models() {
    echo "Checking model files..."
    
    MISTRAL_PATH="$PROJECT_ROOT/models/mistral/snapshots/2dcff66eac0c01dc50e4c41eea959968232187fe"
    
    if [ -d "$MISTRAL_PATH" ]; then
        echo "Found Mistral model at: $MISTRAL_PATH"
        
        # Check for required files
        if [ -f "$MISTRAL_PATH/config.json" ] && [ -f "$MISTRAL_PATH/tokenizer.json" ]; then
            echo "Model files verification passed"
        else
            echo "WARNING: Model files may be incomplete"
        fi
    else
        echo "ERROR: Mistral model not found at $MISTRAL_PATH"
        echo "Download from: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1"
        exit 1
    fi
}

# Function to check configuration files
check_config() {
    echo "Checking configuration files..."
    
    CONFIG_DIR="$PROJECT_ROOT/configs"
    mkdir -p "$CONFIG_DIR"
    
    CLOUD_CONFIG="$CONFIG_DIR/cloud_config.json"
    
    if [ ! -f "$CLOUD_CONFIG" ]; then
        echo "Creating default cloud configuration..."
        cat > "$CLOUD_CONFIG" << EOF
{
    "model_path": "models/mistral/snapshots/2dcff66eac0c01dc50e4c41eea959968232187fe",
    "max_concurrent_tasks": 32,
    "tasks_per_second": 8.0,
    "memory_gb": 256,
    "gpu_memory_gb": 11,
    "preemption_threshold": 1.5,
    "max_batch_size": 16,
    "min_batch_size": 4,
    "max_new_tokens": 100,
    "temperature": 0.7,
    "server_port": 8000,
    "log_level": "INFO"
}
EOF
        echo "Created default configuration: $CLOUD_CONFIG"
    else
        echo "Found existing configuration: $CLOUD_CONFIG"
    fi
}

# Function to start cloud server
start_server() {
    echo "Starting cloud server..."
    
    cd "$PROJECT_ROOT"
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Set Python path
    export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"
    
    # Create logs directory
    mkdir -p logs
    
    # Start server
    CONFIG_FILE="configs/cloud_config.json"
    LOG_FILE="logs/cloud_server_$(date +%Y%m%d_%H%M%S).log"
    
    echo "Starting cloud server with config: $CONFIG_FILE"
    echo "Logs will be written to: $LOG_FILE"
    
    python3 src/cloud_server.py --config "$CONFIG_FILE" 2>&1 | tee "$LOG_FILE"
}

# Function to run health check
health_check() {
    echo "Running health check..."
    
    cd "$PROJECT_ROOT"
    source venv/bin/activate
    export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"
    
    python3 -c "
import sys
sys.path.append('src')
from cloud_server import CloudServer
import asyncio
import json

async def test():
    try:
        server = CloudServer('configs/cloud_config.json')
        await server.initialize()
        stats = server.get_server_stats()
        print('Health check passed:')
        print(json.dumps(stats, indent=2))
        return True
    except Exception as e:
        print(f'Health check failed: {e}')
        return False

result = asyncio.run(test())
sys.exit(0 if result else 1)
"
    
    if [ $? -eq 0 ]; then
        echo "Health check PASSED"
    else
        echo "Health check FAILED"
        exit 1
    fi
}

# Main deployment function
deploy() {
    echo "Starting LL-CIoT cloud server deployment..."
    
    check_dependencies
    check_hardware
    setup_environment
    check_models
    check_config
    
    if [ "$1" = "--health-check-only" ]; then
        health_check
        echo "Health check complete"
        exit 0
    fi
    
    if [ "$1" = "--daemon" ]; then
        echo "Starting server in daemon mode..."
        nohup bash "$0" --start-server > logs/daemon.log 2>&1 &
        echo "Server started in background. PID: $!"
        echo "Check logs: tail -f logs/daemon.log"
    elif [ "$1" = "--start-server" ]; then
        start_server
    else
        echo "Deployment setup complete!"
        echo ""
        echo "Next steps:"
        echo "  1. Health check: $0 --health-check-only"
        echo "  2. Start server:  $0 --start-server"
        echo "  3. Start daemon:  $0 --daemon"
        echo ""
    fi
}

# Parse command line arguments
case "$1" in
    --health-check-only)
        check_dependencies
        setup_environment
        health_check
        ;;
    --start-server)
        deploy --start-server
        ;;
    --daemon)
        deploy --daemon
        ;;
    *)
        deploy
        ;;
esac