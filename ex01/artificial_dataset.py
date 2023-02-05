import csv
import numpy as np

# number of data points
n = 300

# Generate 6 columns with different means and standard deviations
col1 = np.random.normal(loc=0, scale=1, size=n)
col2 = np.random.normal(loc=2, scale=3, size=n)
col3 = np.random.normal(loc=5, scale=2, size=n)
col4 = np.random.normal(loc=10, scale=4, size=n)
col5 = np.random.normal(loc=2.5, scale=1, size=n)
col6 = np.random.randint(low=0, high=10, size=n)

# convert col6 to float
col6 = col6.astype(float)

# Combine the columns into a 2D array
data = np.column_stack((col1, col2, col3, col4, col5, col6))

# Positively correlated columns
data[:, 0] = data[:, 0] + data[:, 2]

# Negatively correlated columns
data[:, 1] = data[:, 1] - data[:, 3]

# Correlation close to 0
data[:, 4] = data[:, 4] + np.random.normal(loc=0, scale=0.5, size=n)

# Save the generated data to a csv file
with open('artificial_dataset.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(data)
