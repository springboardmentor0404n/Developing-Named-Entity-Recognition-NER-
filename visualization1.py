import matplotlib.pyplot as plt
import numpy as np

# Company names
companies = ['Apple', 'Tesla', 'Pfizer', 'Amazon', 'Meta', 'Microsoft', 'Google']

# Metrics
revenues = [416, 94, 81, 524, 134, 618, 343]  # in billions USD
growth = [15, 12, 14, 11, 13, 10, 9]          # in %
reporting_freq = [2, 1, 1, 2, 1, 2, 1]        # number of quarters reported

# Bar positions
x = np.arange(len(companies))
width = 0.25

# Create the grouped bar chart
plt.figure(figsize=(12, 6))
bar1 = plt.bar(x - width, revenues, width, label='Revenue ($B)', color='cornflowerblue')
bar2 = plt.bar(x, growth, width, label='Growth (%)', color='mediumseagreen')
bar3 = plt.bar(x + width, reporting_freq, width, label='Reporting Frequency', color='salmon')

# Add value labels
def annotate_bars(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 2, f'{height}', ha='center', fontsize=9)

annotate_bars(bar1)
annotate_bars(bar2)
annotate_bars(bar3)

# Styling
plt.title('Financial Metrics Comparison by Company', fontsize=14)
plt.xticks(x, companies, rotation=45)
plt.ylabel('Values')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Show the chart
plt.show()
