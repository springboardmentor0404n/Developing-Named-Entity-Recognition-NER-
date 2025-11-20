import matplotlib.pyplot as plt

# Company names and their percentage growth values
companies = ['Apple', 'Tesla', 'Pfizer', 'Amazon', 'Meta', 'Microsoft', 'Google']
growth_percentages = [15, 12, 14, 11, 13, 10, 9]  # in %

# Create the bar chart
plt.style.use('seaborn-v0_8')
plt.figure(figsize=(10, 6))
bars = plt.bar(companies, growth_percentages, color='skyblue', edgecolor='black')

# Annotate bars with percentage values
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, f'{height}%', ha='center', va='bottom', fontsize=10)

# Chart styling
plt.title('Percentage Growth by Company', fontsize=14)
plt.xlabel('Company', fontsize=12)
plt.ylabel('Growth (%)', fontsize=12)
plt.ylim(0, max(growth_percentages) + 5)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Show the chart
plt.show()
