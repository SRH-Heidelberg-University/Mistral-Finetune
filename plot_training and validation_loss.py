import matplotlib.pyplot as plt
import pandas as pd

# Read the Excel file
log_data = pd.read_excel('results-1.xlsx')

# Plotting
plt.plot(log_data['Step'], log_data['Training Loss'], label="Training Loss", marker='o')
plt.plot(log_data['Step'], log_data['Validation Loss'], label="Validation Loss", marker='o')
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss Over Time")
plt.show()

