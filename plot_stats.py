#!/usr/bin/env python3
import sys
import os
import csv
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
                    divergences.append(float(row.get('divergence', 0.0)))
                    weights.append(float(row.get('heuristic_weight', 0.0)))
                except (ValueError, KeyError):
                    # Skip incomplete or corrupt lines written concurrently
                    continue
    except Exception:
        # File might be locked or concurrently written to
        pass
        
    return games, losses, moves, times, nps, divergences, weights

def main():
    csv_path = "checkpoints/stats.csv"
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        if os.path.isdir(csv_path):
            csv_path = os.path.join(csv_path, "stats.csv")
            
    print(f"Monitoring stats in: {csv_path}")
    print("Press Ctrl+C in the terminal to stop.")
    
    # Set up matplotlib plotting style
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Causal Chess Training Live Stats", fontsize=16, fontweight='bold')
    
    # Prevent tight layout warnings
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    def update(frame):
        games, losses, moves, times, nps, divergences, weights = read_stats(csv_path)
        if not games:
            return
            
        # Subplot 1: Average TD Loss
        axs[0, 0].clear()
        axs[0, 0].plot(games, losses, color='#1f77b4', linewidth=1.5)
        axs[0, 0].set_title("Average TD Loss")
        axs[0, 0].set_xlabel("Game")
        axs[0, 0].set_ylabel("Loss")
        
        # Subplot 2: Average H-NN Divergence
        axs[0, 1].clear()
        axs[0, 1].plot(games, divergences, color='#ff7f0e', linewidth=1.5)
        axs[0, 1].set_title("Heuristic - NN Divergence (MAE)")
        axs[0, 1].set_xlabel("Game")
        axs[0, 1].set_ylabel("Divergence")
        
        # Subplot 3: Heuristic Weight (w)
        axs[1, 0].clear()
        axs[1, 0].plot(games, weights, color='#2ca02c', linewidth=1.5)
        axs[1, 0].set_title("Heuristic Weight (w) Monotonic Decay")
        axs[1, 0].set_xlabel("Game")
        axs[1, 0].set_ylabel("Weight")
        axs[1, 0].set_ylim(-0.05, 1.05)
        
        # Subplot 4: Moves per Game
        axs[1, 1].clear()
        axs[1, 1].plot(games, moves, color='#d62728', linewidth=1.5)
        axs[1, 1].set_title("Moves per Game")
        axs[1, 1].set_xlabel("Game")
        axs[1, 1].set_ylabel("Moves")
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    ani = FuncAnimation(fig, update, interval=2000, cache_frame_data=False)
    plt.show()

if __name__ == '__main__':
    main()
