import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

def plot_bioactive_synchronization(log_file):
    epochs = []
    sagnac_errors = []
    langevin_heats = []
    
    with open(log_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            epochs.append(data['epoch'])
            sagnac_errors.append(data['sagnac_error'])
            langevin_heats.append(data['langevin_heat'])
            
    # Phase Synchronization Order Parameter (R) is roughly 1 - Sagnac Error
    # Since Sagnac = 1 - R, R = 1 - Sagnac
    order_parameters = [1.0 - err for err in sagnac_errors]
    
    # Calculate a moving average of the Sagnac Error to represent the Dynamic Threshold
    window = 5
    dynamic_threshold = np.convolve(sagnac_errors, np.ones(window)/window, mode='valid')
    # Pad to match length
    dynamic_threshold = np.pad(dynamic_threshold, (window-1, 0), 'edge')

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Epoch (Time Step)')
    ax1.set_ylabel('Order Parameter (R)', color=color)
    ax1.plot(epochs, order_parameters, color=color, label='Phase Synchronization (R)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim([0, 1.05])

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Sagnac Error / Heat', color=color)  
    ax2.plot(epochs, sagnac_errors, color='tab:orange', alpha=0.5, label='Raw Sagnac Error')
    ax2.plot(epochs, dynamic_threshold, color='tab:red', linestyle='--', label='Dynamic Sagnac Threshold')
    ax2.plot(epochs, langevin_heats, color='tab:purple', alpha=0.3, label='Langevin Heat (Simmer vs Shock)')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  
    plt.title(f"Bioactive Thermodynamic Synchronization: {os.path.basename(log_file)}")
    
    # Add legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')
    
    out_file = log_file.replace('.jsonl', '_bioactive_plot.png')
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {out_file}")

if __name__ == '__main__':
    # Find the most recent telemetry log
    logs = glob.glob('telemetry_logs/*.jsonl')
    if not logs:
        print("No telemetry logs found.")
    else:
        latest_log = max(logs, key=os.path.getctime)
        print(f"Plotting {latest_log}...")
        plot_bioactive_synchronization(latest_log)
