#!/usr/bin/env python3
import sys
import os
import csv
import time

# Try to detect if we have a GUI display. If not, or if matplotlib raises an error on import/show,
# fallback to saving plot images directly to file.
gui_available = True
if sys.platform == 'darwin':
    # On macOS, check if we are in a headless SSH session
    if 'SSH_CLIENT' in os.environ or 'SSH_TTY' in os.environ:
        gui_available = False
elif sys.platform.startswith('linux'):
    # On Linux, check DISPLAY
    if 'DISPLAY' not in os.environ:
        gui_available = False

if not gui_available:
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def read_stats(csv_path):
    games = []
    losses = []
    moves = []
    times = []
    nps = []
    divergences = []
    weights = []
    
    if not os.path.exists(csv_path):
        return games, losses, moves, times, nps, divergences, weights
        
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    games.append(int(row['game']))
                    losses.append(float(row['loss']))
                    moves.append(int(row['moves']))
                    times.append(float(row['time']))
                    nps.append(int(row['nps']))
                    
                    div_val = row.get('divergence')
                    divergences.append(float(div_val) if (div_val is not None and div_val.strip() != "") else 0.0)
                    
                    w_val = row.get('heuristic_weight')
                    weights.append(float(w_val) if (w_val is not None and w_val.strip() != "") else 0.0)
                except (ValueError, KeyError):
                    # Skip incomplete or corrupt lines written concurrently
                    continue
    except Exception:
        # File might be locked or concurrently written to
        pass
        
    return games, losses, moves, times, nps, divergences, weights

def moving_average(data, window_size=20):
    if not data:
        return []
    ma = []
    for i in range(len(data)):
        start = max(0, i - window_size + 1)
        window = data[start:i+1]
        ma.append(sum(window) / len(window))
    return ma

def main():
    csv_path = "checkpoints/stats.csv"
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        if os.path.isdir(csv_path):
            csv_path = os.path.join(csv_path, "stats.csv")
    else:
        # Default auto-resolve: if checkpoints/stats.csv does not exist,
        # but ../checkpoints/stats.csv does, use the latter.
        if not os.path.exists(csv_path) and os.path.exists("../checkpoints/stats.csv"):
            csv_path = "../checkpoints/stats.csv"

    # Derive output PNG path in the same directory as the CSV
    dir_name = os.path.dirname(csv_path) or "."
    base_name = os.path.basename(csv_path)
    png_name = os.path.splitext(base_name)[0] + ".png"
    output_png_path = os.path.join(dir_name, png_name)
            
    print(f"Monitoring stats in: {csv_path}", flush=True)
    print(f"Saving plot image to: {output_png_path}", flush=True)
    if gui_available:
        print("GUI display detected. Interactive window will open.", flush=True)
    else:
        print("Running in headless mode (no GUI display detected). Plots will be saved to file.", flush=True)
    print("Press Ctrl+C in the terminal to stop.\n", flush=True)
    
    # Set up matplotlib plotting style
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Causal Chess Training Live Stats", fontsize=16, fontweight='bold')
    
    # Prevent tight layout warnings
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    last_num_games = -1
    file_warned = False
    
    def update(frame):
        nonlocal last_num_games, file_warned
        
        if not os.path.exists(csv_path):
            if not file_warned:
                print(f"Warning: Log file not found at {csv_path}. Waiting for it to be created...", flush=True)
                file_warned = True
            return
            
        games, losses, moves, times, nps, divergences, weights = read_stats(csv_path)
        if not games:
            if not file_warned:
                print(f"Log file {csv_path} is empty or formatting is incorrect. Waiting for data...", flush=True)
                file_warned = True
            return
            
        if file_warned:
            print(f"Active data found in {csv_path}. Starting updates.", flush=True)
            file_warned = False
            
        num_games = len(games)
        if num_games == last_num_games:
            return
            
        last_num_games = num_games
        
        # Console output of latest metrics
        print(f"[Update] Games: {num_games} | Latest: Loss={losses[-1]:.6f}, Moves={moves[-1]}, NPS={nps[-1]}, Divergence={divergences[-1]:.4f}, Weight={weights[-1]:.4f}", flush=True)
        
        # Subplot 1: Average TD Loss
        axs[0, 0].clear()
        axs[0, 0].plot(games, losses, color='#1f77b4', alpha=0.3, linewidth=1.0, label='Raw')
        axs[0, 0].plot(games, moving_average(losses, 20), color='#1f77b4', linewidth=2.0, label='MA (20)')
        axs[0, 0].set_title("Average TD Loss")
        axs[0, 0].set_xlabel("Game")
        axs[0, 0].set_ylabel("Loss")
        axs[0, 0].legend()
        
        # Subplot 2: Average H-NN Divergence
        axs[0, 1].clear()
        axs[0, 1].plot(games, divergences, color='#ff7f0e', alpha=0.3, linewidth=1.0, label='Raw')
        axs[0, 1].plot(games, moving_average(divergences, 20), color='#ff7f0e', linewidth=2.0, label='MA (20)')
        axs[0, 1].set_title("Heuristic - NN Divergence (MAE)")
        axs[0, 1].set_xlabel("Game")
        axs[0, 1].set_ylabel("Divergence")
        axs[0, 1].legend()
        
        # Subplot 3: Heuristic Weight (w)
        axs[1, 0].clear()
        axs[1, 0].plot(games, weights, color='#2ca02c', alpha=0.3, linewidth=1.0, label='Raw')
        axs[1, 0].plot(games, moving_average(weights, 20), color='#2ca02c', linewidth=2.0, label='MA (20)')
        axs[1, 0].set_title("Heuristic Weight (w) Monotonic Decay")
        axs[1, 0].set_xlabel("Game")
        axs[1, 0].set_ylabel("Weight")
        axs[1, 0].set_ylim(-0.05, 1.05)
        axs[1, 0].legend()
        
        # Subplot 4: Moves per Game
        axs[1, 1].clear()
        axs[1, 1].plot(games, moves, color='#d62728', alpha=0.3, linewidth=1.0, label='Raw')
        axs[1, 1].plot(games, moving_average(moves, 20), color='#d62728', linewidth=2.0, label='MA (20)')
        axs[1, 1].set_title("Moves per Game")
        axs[1, 1].set_xlabel("Game")
        axs[1, 1].set_ylabel("Moves")
        axs[1, 1].legend()
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save plot image
        try:
            plt.savefig(output_png_path)
        except Exception as e:
            print(f"Error saving plot to {output_png_path}: {e}", flush=True)

    if gui_available:
        ani = FuncAnimation(fig, update, interval=2000, cache_frame_data=False)
        plt.show()
    else:
        try:
            while True:
                update(None)
                time.sleep(2)
        except KeyboardInterrupt:
            print("\nStopped monitoring.", flush=True)
            sys.exit(0)

if __name__ == '__main__':
    main()
