#!/usr/bin/env python3

import os
import subprocess
import datetime
import json

def collect_logs(output_dir, max_lines=10000):
    """Collect system logs for training data"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Define log sources
    log_sources = [
        {'name': 'syslog', 'command': f'tail -n {max_lines} /var/log/syslog'},
        {'name': 'auth', 'command': f'tail -n {max_lines} /var/log/auth.log'},
        {'name': 'kernel', 'command': 'dmesg | tail -n 1000'}
    ]
    
    # Get timestamp for filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Collect logs from each source
    for source in log_sources:
        try:
            result = subprocess.run(
                source['command'], 
                shell=True, 
                capture_output=True, 
                text=True
            )
            
            if result.returncode == 0:
                output_file = os.path.join(output_dir, f"{source['name']}_{timestamp}.txt")
                
                with open(output_file, 'w') as f:
                    f.write(result.stdout)
                    
                print(f"Collected {source['name']} logs to {output_file}")
            else:
                print(f"Error collecting {source['name']} logs: {result.stderr}")
        except Exception as e:
            print(f"Exception collecting {source['name']} logs: {e}")
    
    print(f"Log collection complete. Files saved to {output_dir}")

if __name__ == "__main__":
    # Directory for training data
    data_dir = "../data/log_analyzer/training"
    collect_logs(data_dir)
