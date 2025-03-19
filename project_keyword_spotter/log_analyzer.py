#!/usr/bin/env python3

import os
import time
import json
import re
import subprocess
import numpy as np
import pickle
import hashlib
from collections import Counter
from datetime import datetime
from typing import List, Dict, Set, Tuple, Optional, Any, Union
import click

# Configure paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(BASE_DIR, 'config/log_analyzer')
MODEL_DIR = os.path.join(BASE_DIR, 'models/log_analyzer')
DATA_DIR = os.path.join(BASE_DIR, 'data/log_analyzer')

CONFIG_PATH = os.path.join(CONFIG_DIR, 'config.json')
MODEL_PATH = os.path.join(MODEL_DIR, 'syslog_anomaly_detector_edgetpu.tflite')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'log_vectorizer.pkl')
LOG_HISTORY_PATH = os.path.join(DATA_DIR, 'log_history.txt')
ANOMALIES_DIR = os.path.join(DATA_DIR, 'anomalies')

# Ensure directories exist
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ANOMALIES_DIR, exist_ok=True)

# Default configuration
DEFAULT_CONFIG = {
    "log_paths": ["/var/log/syslog", "/var/log/auth.log"],
    "anomaly_threshold": 0.8,
    "max_logs_per_check": 1000
}

# Vectorizer functions
def tokenize(text: str) -> List[str]:
    """Simple tokenization by splitting on whitespace and punctuation"""
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    return text.split()

def extract_patterns(text: str) -> List[int]:
    """Extract common log patterns"""
    # IP addresses
    has_ip = 1 if re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', text) is not None else 0
    
    # Process IDs
    has_pid = 1 if re.search(r'\bpid[=:]\d+\b', text) is not None else 0
    
    # User references
    has_user = 1 if re.search(r'\buser\b|\busername\b|\blogin\b', text) is not None else 0
    
    # Error or warning indicators
    has_error = 1 if re.search(r'\berror\b|\bfail\b|\bfailed\b', text, re.I) is not None else 0
    has_warning = 1 if re.search(r'\bwarn\b|\bwarning\b', text, re.I) is not None else 0
    
    return [has_ip, has_pid, has_user, has_error, has_warning]

def fit_vectorizer(texts: List[str], max_features: int = 100) -> Dict:
    """Build a vectorizer from texts"""
    all_tokens = []
    for text in texts:
        all_tokens.extend(tokenize(text))
    
    # Count token frequencies
    token_counts = Counter(all_tokens)
    
    # Select the most common tokens as features
    most_common = token_counts.most_common(max_features)
    
    # Create the feature map
    feature_map = {token: idx for idx, (token, _) in enumerate(most_common)}
    return {'feature_map': feature_map, 'is_fitted': True}

def transform_texts(vectorizer_data: Dict, texts: List[str]) -> np.ndarray:
    """Transform texts to feature vectors"""
    if not vectorizer_data.get('is_fitted', False):
        raise ValueError("Vectorizer is not fitted")
    
    feature_map = vectorizer_data['feature_map']
    feature_vectors = []
    
    for text in texts:
        # Get token counts
        tokens = tokenize(text)
        token_counts = Counter(tokens)
        
        # Create vector from token counts
        vector = np.zeros(len(feature_map) + 5, dtype=np.int8)
        for token, count in token_counts.items():
            if token in feature_map:
                vector[feature_map[token]] = min(count, 127)  # Cap at int8 max
        
        # Add pattern features
        patterns = extract_patterns(text)
        vector[-5:] = patterns
        
        feature_vectors.append(vector)
    
    return np.array(feature_vectors)

def save_vectorizer(vectorizer_data: Dict, path: str) -> None:
    """Save vectorizer data to file"""
    with open(path, 'wb') as f:
        pickle.dump(vectorizer_data, f)

def load_vectorizer(path: str) -> Dict:
    """Load vectorizer data from file"""
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError, pickle.PickleError) as e:
        print(f"Error loading vectorizer: {e}")
        print("Creating a new vectorizer")
        return {'feature_map': {}, 'is_fitted': False}

# Log processing functions
def preprocess_log(log_entry: str) -> str:
    """Preprocess a log entry before vectorization"""
    # Remove timestamps
    log_entry = re.sub(r'^\w+\s+\d+\s+\d+:\d+:\d+', '', log_entry)
    
    # Remove IP addresses
    log_entry = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', 'IP_ADDR', log_entry)
    
    # Remove specific numbers
    log_entry = re.sub(r'\b\d+\b', 'NUM', log_entry)
    
    return log_entry.strip()

def get_log_hash(log_entry: str) -> str:
    """Generate a hash for a log entry to avoid duplicates"""
    return hashlib.md5(log_entry.encode()).hexdigest()

def get_recent_logs(log_paths: List[str], max_logs: int = 1000) -> List[str]:
    """Get recent log entries from specified log files"""
    log_entries = []
    
    for log_path in log_paths:
        try:
            cmd = f"tail -n {max_logs} {log_path}"
            
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                entries = result.stdout.strip().split('\n')
                log_entries.extend([e for e in entries if e.strip()])
            else:
                print(f"Error reading log {log_path}: {result.stderr}")
        except Exception as e:
            print(f"Exception reading log {log_path}: {e}")
    
    return log_entries

def load_log_history() -> Set[str]:
    """Load the history of processed logs"""
    try:
        with open(LOG_HISTORY_PATH, 'r') as f:
            return set(line.strip() for line in f)
    except FileNotFoundError:
        return set()

def save_log_history(log_hashes: Set[str]) -> None:
    """Save the history of processed logs"""
    with open(LOG_HISTORY_PATH, 'w') as f:
        for log_hash in log_hashes:
            f.write(f"{log_hash}\n")

def filter_new_logs(log_entries: List[str], log_hashes: Set[str]) -> Tuple[List[str], Set[str]]:
    """Filter out logs we've already processed"""
    new_logs = []
    updated_hashes = log_hashes.copy()
    
    for entry in log_entries:
        if not entry.strip():
            continue
            
        log_hash = get_log_hash(entry)
        if log_hash not in updated_hashes:
            new_logs.append(entry)
            updated_hashes.add(log_hash)
    
    # Keep the hash set from growing too large
    if len(updated_hashes) > 10000:
        updated_hashes = set(list(updated_hashes)[-10000:])
    
    return new_logs, updated_hashes

# Model and anomaly detection
def load_edge_tpu_model(model_path: str) -> Optional[Any]:
    """Load the TFLite model with Edge TPU if available"""
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Will run in collection-only mode")
        return None
        
    try:
        from tflite_runtime.interpreter import Interpreter, load_delegate
        
        try:
            # Try to load with Edge TPU acceleration
            delegates = [load_delegate('libedgetpu.so.1')]
            interpreter = Interpreter(
                model_path=model_path,
                experimental_delegates=delegates)
            print("Loaded model with Edge TPU acceleration")
        except Exception as e:
            print(f"Failed to load model with Edge TPU: {e}")
            print("Trying CPU fallback...")
            interpreter = Interpreter(model_path=model_path)
            print("Loaded model for CPU inference")
        
        interpreter.allocate_tensors()
        return interpreter
    except ImportError:
        print("TFLite Runtime not available. Running in collection-only mode.")
        return None

def detect_keyword_anomalies(log_entries: List[str]) -> List[Dict]:
    """Detect anomalies in logs using keyword matching"""
    anomalies = []
    
    for entry in log_entries:
        if re.search(r'\berror\b|\bfail\b|\bwarning\b|\bcritical\b|\balert\b', 
                    entry, re.IGNORECASE):
            anomalies.append({
                'entry': entry,
                'confidence': 1.0,
                'reason': 'keyword_match'
            })
    
    return anomalies

def detect_model_anomalies(
    interpreter: Any, 
    vectorizer_data: Dict, 
    log_entries: List[str],
    threshold: float = 0.8
) -> List[Dict]:
    """Detect anomalies using the TFLite model"""
    if not log_entries:
        return []
        
    # Preprocess the logs
    preprocessed_logs = [preprocess_log(entry) for entry in log_entries]
    
    # Vectorize the logs
    try:
        features = transform_texts(vectorizer_data, preprocessed_logs)
    except ValueError as e:
        print(f"Error vectorizing logs: {e}")
        return []
    
    anomalies = []
    
    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    for i, (features_vector, entry) in enumerate(zip(features, log_entries)):
        # Reshape for model input
        input_data = np.expand_dims(features_vector, axis=0)
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference
        interpreter.invoke()
        
        # Get output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        anomaly_score = float(np.squeeze(output_data))
        
        # Check if it's an anomaly
        if anomaly_score > threshold:
            anomalies.append({
                'entry': entry,
                'confidence': anomaly_score,
                'reason': 'model_prediction'
            })
    
    return anomalies

def save_anomalies(anomalies: List[Dict]) -> None:
    """Save detected anomalies to a file"""
    if not anomalies:
        return
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    anomaly_file = os.path.join(ANOMALIES_DIR, f"anomalies_{timestamp}.txt")
    
    with open(anomaly_file, 'w') as f:
        f.write(f"Anomalies detected at {datetime.now()}\n")
        f.write("=" * 80 + "\n\n")
        
        for anomaly in anomalies:
            f.write(f"Confidence: {anomaly['confidence']:.4f}, Reason: {anomaly['reason']}\n")
            f.write(f"Log: {anomaly['entry']}\n\n")
    
    print(f"Saved {len(anomalies)} anomalies to {anomaly_file}")

def collect_training_data(log_entries: List[str], sample_size: int = 100) -> None:
    """Collect logs for future training"""
    if not log_entries or len(log_entries) < 10:
        return
        
    # Save a subset of logs for training
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_file = os.path.join(DATA_DIR, f"training_logs_{timestamp}.txt")
    
    with open(training_file, 'w') as f:
        for entry in log_entries[:min(sample_size, len(log_entries))]:
            f.write(f"{entry}\n")
    
    print(f"Saved {min(sample_size, len(log_entries))} logs for training to {training_file}")

def load_config() -> Dict:
    """Load configuration from file or create default"""
    try:
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Config file not found or invalid. Creating default at {CONFIG_PATH}")
        with open(CONFIG_PATH, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        return DEFAULT_CONFIG

# Main functions
def analyze_logs(verbose: bool = False) -> None:
    """Main function to analyze logs for anomalies"""
    if verbose:
        print(f"Starting log analysis at {datetime.now()}")
    
    # Load configuration
    config = load_config()
    
    # Load model and vectorizer
    interpreter = load_edge_tpu_model(MODEL_PATH)
    vectorizer_data = load_vectorizer(VECTORIZER_PATH)
    
    # Load log history and get recent logs
    log_hashes = load_log_history()
    log_entries = get_recent_logs(
        config.get('log_paths', ['/var/log/syslog']),
        config.get('max_logs_per_check', 1000)
    )
    
    if verbose:
        print(f"Found {len(log_entries)} log entries")
    
    # Filter out logs we've already seen
    new_logs, updated_log_hashes = filter_new_logs(log_entries, log_hashes)
    
    if verbose:
        print(f"Found {len(new_logs)} new log entries")
    
    if not new_logs:
        if verbose:
            print("No new logs to analyze")
        return
    
    # Detect anomalies
    if interpreter is not None and vectorizer_data.get('is_fitted', False):
        # Use model for detection
        anomalies = detect_model_anomalies(
            interpreter, 
            vectorizer_data, 
            new_logs,
            config.get('anomaly_threshold', 0.8)
        )
    else:
        # Use keyword matching
        anomalies = detect_keyword_anomalies(new_logs)
    
    # Save results
    if anomalies:
        if verbose:
            print(f"Detected {len(anomalies)} anomalies")
        save_anomalies(anomalies)
    elif verbose:
        print("No anomalies detected")
    
    # Collect some data for training
    collect_training_data(new_logs)
    
    # Save log history
    save_log_history(updated_log_hashes)
    
    if verbose:
        print(f"Analysis completed at {datetime.now()}")

def initialize_vectorizer() -> None:
    """Initialize an empty vectorizer file"""
    vectorizer_data = {'feature_map': {}, 'is_fitted': False}
    save_vectorizer(vectorizer_data, VECTORIZER_PATH)
    print(f"Created initial vectorizer at {VECTORIZER_PATH}")

def train_vectorizer(sample_size: int = 1000) -> None:
    """Train the vectorizer using collected log data"""
    # Find all training files
    training_files = [
        os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR)
        if f.startswith('training_logs_') and f.endswith('.txt')
    ]
    
    if not training_files:
        print("No training files found")
        return
    
    print(f"Found {len(training_files)} training files")
    
    # Load log data
    all_logs = []
    for file_path in training_files:
        try:
            with open(file_path, 'r') as f:
                logs = [line.strip() for line in f if line.strip()]
                all_logs.extend(logs)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    if not all_logs:
        print("No log data found in training files")
        return
    
    print(f"Loaded {len(all_logs)} log entries for training")
    
    # Preprocess logs
    preprocessed_logs = [preprocess_log(entry) for entry in all_logs]
    
    # Fit vectorizer (use a sample if there are too many logs)
    if len(preprocessed_logs) > sample_size:
        import random
        sample_logs = random.sample(preprocessed_logs, sample_size)
    else:
        sample_logs = preprocessed_logs
    
    print(f"Training vectorizer with {len(sample_logs)} log entries")
    vectorizer_data = fit_vectorizer(sample_logs)
    
    # Save vectorizer
    save_vectorizer(vectorizer_data, VECTORIZER_PATH)
    print(f"Vectorizer trained and saved to {VECTORIZER_PATH}")

# CLI interface
@click.group()
def cli():
    """Analyze system logs for anomalies using Edge TPU."""
    pass

@cli.command()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def analyze(verbose):
    """Analyze logs for anomalies."""
    analyze_logs(verbose)

@cli.command()
def init():
    """Initialize the log analyzer (create vectorizer and config)."""
    initialize_vectorizer()
    load_config()  # This will create default config if not exists
    print("Log analyzer initialized")

@cli.command()
@click.option('--sample-size', '-s', type=int, default=1000, 
              help='Number of log entries to use for training')
def train(sample_size):
    """Train the vectorizer using collected log data."""
    train_vectorizer(sample_size)

@cli.command()
def status():
    """Show status of the log analyzer components."""
    # Check config
    config_exists = os.path.exists(CONFIG_PATH)
    if config_exists:
        print(f"✓ Config file exists at {CONFIG_PATH}")
    else:
        print(f"✗ Config file not found at {CONFIG_PATH}")
    
    # Check vectorizer
    vectorizer_exists = os.path.exists(VECTORIZER_PATH)
    if vectorizer_exists:
        try:
            vectorizer_data = load_vectorizer(VECTORIZER_PATH)
            is_fitted = vectorizer_data.get('is_fitted', False)
            feature_count = len(vectorizer_data.get('feature_map', {}))
            
            if is_fitted:
                print(f"✓ Vectorizer is fitted with {feature_count} features")
            else:
                print("✓ Vectorizer exists but is not fitted")
        except Exception as e:
            print(f"✗ Error reading vectorizer: {e}")
    else:
        print(f"✗ Vectorizer not found at {VECTORIZER_PATH}")
    
    # Check model
    model_exists = os.path.exists(MODEL_PATH)
    if model_exists:
        print(f"✓ Model file exists at {MODEL_PATH}")
    else:
        print(f"✗ Model file not found at {MODEL_PATH}")
    
    # Check training data
    training_files = [
        f for f in os.listdir(DATA_DIR)
        if f.startswith('training_logs_') and f.endswith('.txt')
    ]
    print(f"Found {len(training_files)} training files in {DATA_DIR}")
    
    # Check anomalies
    anomaly_files = os.listdir(ANOMALIES_DIR)
    print(f"Found {len(anomaly_files)} anomaly files in {ANOMALIES_DIR}")

if __name__ == "__main__":
    cli()
