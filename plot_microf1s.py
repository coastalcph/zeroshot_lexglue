import matplotlib.pyplot as plt
import numpy as np

# models = ("Random Guess", "Biased Guess", "Zero-shot GPT-T", 'Bert Reported')
# models_means = {
#     'Random Guess': (3.3),
#     'Biased Guess': (4.8),
#     'Zero-shot GPT-T': (47.6),
#     'Bert Reported': (78.9)
#
# }
#
# x = np.arange(len(models))  # the label locations
# width = 0.25  # the width of the bars
# multiplier = 0
#
# fig, ax = plt.subplots(layout='constrained')
#
# for model in models:
#     offset = width * multiplier
#     rects = ax.bar(x + offset, models_means[model], width, label=model)
#     ax.bar_label(rects, padding=3)
#     multiplier += 1




import matplotlib.pyplot as plt

# Define the data for the bars
heights = [3.3, 4.8, 49.0, 78.9]
colors = ['red', 'orange', 'blue', 'green']
models = ("Random Guess", "Biased Guess", "Zero-shot ChatGPT", 'Best Reported')
fig, ax = plt.subplots(figsize=(7, 3))
# fig = plt.figure(figsize=(6, 3))

# Create the bar plot
plt.bar(models, heights, color=colors, width=0.7, zorder=2)

# Increase the space between the bars
ax.set_xticks(models)
ax.set_xticklabels(models)
plt.subplots_adjust(wspace=0.7)

# Add a title and axis labels
plt.ylabel('Micro-F1')

plt.ylim(0, 80)
plt.grid(axis='y', zorder=1)

# Show the plot
plt.show()