import matplotlib.pyplot as plt
import os 

def plot_co2_history(dates, co2_arr, baseline_ppm, room_name, tag, save_dir):
    plt.figure(figsize=(12, 6))
    plt.plot(dates, co2_arr, label="CO2 (ppm)", color="blue")
    plt.axhline(y=baseline_ppm, color="red", linestyle="--", label="Baseline CO2 (ppm)")
    plt.title(f"CO2 History for {room_name} ({tag})")
    plt.xlabel("Date")
    plt.ylabel("CO2 (ppm)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, f"{room_name}_{tag}_co2_history.png")
    plt.savefig(plot_path)
    plt.close()

    