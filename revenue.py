import matplotlib.pyplot as plt

# Company names and their revenues in billions USD
companies = ['Apple', 'Tesla', 'Pfizer', 'Amazon', 'Meta', 'Microsoft', 'Google']
revenues = [416, 94, 81, 524, 134, 618, 343]  # in billions

# Create the bar chart
plt.style.use('seaborn-v0_8')
plt.figure(figsize=(10, 6))
bars = plt.bar(companies, revenues, color='cornflowerblue', edgecolor='black')

# Annotate bars with revenue values
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 10, f"${height}B", ha='center', va='bottom', fontsize=10)

# Chart styling
plt.title('Revenue Comparison of Major Companies (in Billions USD)', fontsize=14)
plt.xlabel('Company', fontsize=12)
plt.ylabel('Revenue (USD Billions)', fontsize=12)
plt.ylim(0, max(revenues) + 100)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Show the chart
plt.show()
