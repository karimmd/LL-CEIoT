#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default edge server ID
EDGE_SERVER_ID=${1:-"edge_server_1"}

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
    python3 -c "import torch, transformers, pandas, numpy, psutil" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "ERROR: Missing required Python packages"
        echo "Install with: pip3 install torch transformers pandas numpy psutil"
        exit 1
    fi
    
    echo "Dependencies check passed"
}

# Function to check edge hardware
check_edge_hardware() {
    echo "Checking edge hardware configuration..."
    
    # Check CPU
    CPU_INFO=$(cat /proc/cpuinfo | grep "model name" | head -1 | cut -d: -f2 | xargs)
    echo "CPU: $CPU_INFO"
    
    # Check CPU cores
    CPU_CORES=$(nproc)
    echo "CPU cores: $CPU_CORES"
    if [ "$CPU_CORES" -lt 2 ]; then
        echo "WARNING: CPU cores ($CPU_CORES) may be insufficient for edge deployment"
    fi
    
    # Check system memory
    TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
    echo "System memory: ${TOTAL_MEM}GB"
    if [ "$TOTAL_MEM" -lt 4 ]; then
        echo "WARNING: System memory (${TOTAL_MEM}GB) may be insufficient"
    fi
    
    # Check GPU availability
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU detected:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
        
        # Check GPU memory for edge inference
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        
        if [ "$GPU_MEM" -lt 4000 ]; then
            echo "INFO: Low-end GPU detected - using CPU inference (recommended)"
        fi
    else
        echo "INFO: No GPU detected - using CPU inference"
    fi
    
    # Check disk space for model
    AVAILABLE_SPACE=$(df -BG "$PROJECT_ROOT" | awk 'NR==2 {print $4}' | sed 's/G//')
    echo "Available disk space: ${AVAILABLE_SPACE}GB"
    if [ "$AVAILABLE_SPACE" -lt 5 ]; then
        echo "WARNING: Low disk space (${AVAILABLE_SPACE}GB) - may be insufficient for models"
    fi
}

# Function to setup edge environment
setup_edge_environment() {
    echo "Setting up edge server environment..."
    
    cd "$PROJECT_ROOT"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv_edge" ]; then
        python3 -m venv venv_edge
        echo "Created edge virtual environment"
    fi
    
    # Activate virtual environment
    source venv_edge/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install edge-specific requirements (CPU-optimized)
    pip install torch==2.0.1+cpu torchvision==0.15.2+cpu --index-url https://download.pytorch.org/whl/cpu
    pip install transformers pandas numpy psutil asyncio-throttle
    
    echo "Edge environment setup complete"
}

# Function to check TinyLlama model
check_edge_models() {
    echo "Checking edge model files..."
    
    MODEL_PATH="$PROJECT_ROOT/models/tinyllama/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6"
    
    if [ -d "$MODEL_PATH" ]; then
        echo "Found edge model at: $MODEL_PATH"
        
        # Check for required files
        if [ -f "$MODEL_PATH/config.json" ] && [ -f "$MODEL_PATH/tokenizer.json" ]; then
            echo "Model files verification passed"
            
            # Check model size
            MODEL_SIZE=$(du -sh "$MODEL_PATH" | cut -f1)
            echo "Model size: $MODEL_SIZE"
        else
            echo "WARNING: Model files may be incomplete"
        fi
    else
        echo "ERROR: Edge model not found at $MODEL_PATH"
        echo "Download from: https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        exit 1
    fi
}

# Function to create edge configuration
check_edge_config() {
    echo "Checking edge server configuration..."
    
    CONFIG_DIR="$PROJECT_ROOT/configs"
    mkdir -p "$CONFIG_DIR"
    
    EDGE_CONFIG="$CONFIG_DIR/edge_config.json"
    
    if [ ! -f "$EDGE_CONFIG" ]; then
        echo "Creating default edge configuration..."
        
        # Calculate port based on server ID (8001, 8002, etc.)
        SERVER_NUM=$(echo "$EDGE_SERVER_ID" | grep -o '[0-9]*' | tail -1)
        SERVER_PORT=$((8000 + ${SERVER_NUM:-1}))
        
        cat > "$EDGE_CONFIG" << EOF
{
    "model_path": "models/tinyllama/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6",
    "use_cpu_inference": true,
    "quantization_bits": 8,
    "max_concurrent_tasks": 8,
    "tasks_per_second": 2.0,
    "memory_gb": 16,
    "gpu_memory_gb": 2,
    "cpu_cores": 4,
    "preemption_threshold": 1.2,
    "max_batch_size": 8,
    "min_batch_size": 2,
    "max_new_tokens": 50,
    "temperature": 0.8,
    "server_port": $SERVER_PORT,
    "log_level": "INFO"
}
EOF
        echo "Created default edge configuration: $EDGE_CONFIG"
        echo "Edge server will run on port: $SERVER_PORT"
    else
        echo "Found existing configuration: $EDGE_CONFIG"
    fi
}

# Function to start edge server
start_edge_server() {
    echo "Starting edge server: $EDGE_SERVER_ID..."
    
    cd "$PROJECT_ROOT"
    
    # Activate virtual environment
    source venv_edge/bin/activate
    
    # Set Python path
    export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"
    
    # Create logs directory
    mkdir -p logs
    
    # Start server
    CONFIG_FILE="configs/edge_config.json"
    LOG_FILE="logs/${EDGE_SERVER_ID}_$(date +%Y%m%d_%H%M%S).log"
    
    echo "Starting edge server with config: $CONFIG_FILE"
    echo "Logs will be written to: $LOG_FILE"
    
    # Force CPU inference for edge deployment
    export CUDA_VISIBLE_DEVICES=""
    
    python3 src/edge_server.py \
        --server-id "$EDGE_SERVER_ID" \
        --config "$CONFIG_FILE" \
        2>&1 | tee "$LOG_FILE"
}

# Function to run edge health check
edge_health_check() {
    echo "Running edge server health check..."
    
    cd "$PROJECT_ROOT"
    source venv_edge/bin/activate
    export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"
    export CUDA_VISIBLE_DEVICES=""  # Force CPU for edge
    
    python3 -c "
import sys
sys.path.append('src')
from edge_server import EdgeServer
import asyncio
import json

async def test():
    try:
        server = EdgeServer('$EDGE_SERVER_ID', 'configs/edge_config.json')
        await server.initialize()
        stats = server.get_server_stats()
        print('Edge health check passed:')
        print(json.dumps(stats, indent=2))
        return True
    except Exception as e:
        print(f'Edge health check failed: {e}')
        return False

result = asyncio.run(test())
sys.exit(0 if result else 1)
"
    
    if [ $? -eq 0 ]; then
        echo "Edge health check PASSED"
    else
        echo "Edge health check FAILED"
        exit 1
    fi
}

# Function to optimize for edge deployment
optimize_for_edge() {
    echo "Applying edge deployment optimizations..."
    
    # Set CPU-only environment variables
    export CUDA_VISIBLE_DEVICES=""
    export OMP_NUM_THREADS=$(nproc)
    export MKL_NUM_THREADS=$(nproc)
    
    # Create optimization script
    cat > "$PROJECT_ROOT/set_edge_env.sh" << 'EOF'
#!/bin/bash
# Edge deployment optimization environment
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)
export TOKENIZERS_PARALLELISM=false
echo "Edge optimization environment set (CPU inference)"
EOF
    
    chmod +x "$PROJECT_ROOT/set_edge_env.sh"
    echo "Created edge optimization script: set_edge_env.sh"
}

# Main deployment function
deploy_edge() {
    echo "Starting LL-CIoT edge server deployment..."
    echo "Server ID: $EDGE_SERVER_ID"
    
    check_dependencies
    check_edge_hardware
    setup_edge_environment
    check_edge_models
    check_edge_config
    optimize_for_edge
    
    if [ "$2" = "--health-check-only" ]; then
        edge_health_check
        echo "Edge health check complete"
        exit 0
    fi
    
    if [ "$2" = "--daemon" ]; then
        echo "Starting edge server in daemon mode..."
        nohup bash "$0" "$EDGE_SERVER_ID" --start-server > logs/${EDGE_SERVER_ID}_daemon.log 2>&1 &
        echo "Edge server started in background. PID: $!"
        echo "Check logs: tail -f logs/${EDGE_SERVER_ID}_daemon.log"
    elif [ "$2" = "--start-server" ]; then
        start_edge_server
    else
        echo "Edge deployment setup complete!"
        echo ""
        echo "Next steps:"
        echo "  1. Health check: $0 $EDGE_SERVER_ID --health-check-only"
        echo "  2. Start server:  $0 $EDGE_SERVER_ID --start-server"
        echo "  3. Start daemon:  $0 $EDGE_SERVER_ID --daemon"
        echo ""
        echo "Edge Deployment Optimization:"
        echo "  Source optimization: source set_edge_env.sh"
    fi
}

# Parse command line arguments
case "$2" in
    --health-check-only)
        check_dependencies
        setup_edge_environment
        edge_health_check
        ;;
    --start-server)
        deploy_edge
        ;;
    --daemon)
        deploy_edge
        ;;
    *)
        deploy_edge
        ;;
esac