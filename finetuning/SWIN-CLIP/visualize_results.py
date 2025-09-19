import json
import os
import numpy as np
import matplotlib.pyplot as plt

dir_path = os.path.dirname(os.path.realpath(__file__))
print(F"File path: {dir_path}")

CHECKPOINT_PATH = os.path.join(dir_path, "checkpoints")

checkpoints = os.listdir(CHECKPOINT_PATH)
print(f"Found {len(checkpoints)} checkpoints: {checkpoints}")

results = {checkpoint: {} for checkpoint in checkpoints}

for checkpoint in checkpoints:
    if '100_labels' not in checkpoint:
        continue
    if 'v5' in checkpoint:
        continue

    checkpoint_path = os.path.join(CHECKPOINT_PATH, checkpoint)

    results_json = os.path.join(checkpoint_path, "results.json")
    zeroshot_json = os.path.join(checkpoint_path, "zero_shot_results.json")
    
    if not os.path.exists(results_json):
        print(f"Skipping {checkpoint} as results.json not found")
        continue

    with open(results_json, "r") as f:
        results[checkpoint] = json.load(f)

    del results[checkpoint]['losses']

    if not os.path.exists(zeroshot_json):
        print(f"Skipping {checkpoint} - zero shot results.json not found")
        continue
    
    with open(zeroshot_json, "r") as f:
        results[checkpoint].update(json.load(f))
    
    

# Plotting
num_checkpoints = sum([1 for checkpoint in checkpoints if results[checkpoint] != {}])
checkpoints = sorted([checkpoint for checkpoint in checkpoints if results[checkpoint] != {}])

for c in checkpoints:
    if 'baseline' in c:
        checkpoints.remove(c)
        checkpoints = [c] + checkpoints

print("-----")
print(f"Found results for {num_checkpoints} checkpoints")

print(checkpoints)

plt.figure(figsize=(10, 10))

x = np.arange(len(checkpoints))  # Create evenly spaced x coordinates
width = 0.35  # Width of the bars

for i, checkpoint in enumerate(checkpoints):
    model = checkpoint.split('_')[0]
    version = checkpoint.split('_')[-1]
    if 'baseline' in checkpoint:
        checkpoint_name = 'baseline'
    else:
        checkpoint_name = f"{model} {version}"

    # Plot bars side by side by offsetting their x positions
    final_eval = plt.bar(x[i] - width/2, results[checkpoint]['final_eval_acc'], width, color='teal')
    if 'zero_shot_top1' in results[checkpoint]:
        zero_shot = plt.bar(x[i] + width/2, results[checkpoint]['zero_shot_top1'], width, color='orange')

plt.axvline(x=((x[0]+width/2) + (x[1] - width/2))/2, color='gray', linestyle='--', linewidth=2)
# Set the x-axis labels at the center of each group of bars
plt.xticks(x, [checkpoint.split('_')[0] + ' ' + checkpoint.split('_')[-1] if 'baseline' not in checkpoint 
              else 'baseline' for checkpoint in checkpoints], rotation=45)

plt.legend((final_eval, zero_shot), ['Training Validation', 'Zero Shot'])
plt.xlabel('Model configurations')
plt.ylabel('Top 1 Accuracy (percent)')

plt.savefig(os.path.join(dir_path, "results.png"))










