# Import Library

import numpy as np
import matplotlib.pyplot as plt

# Create figure and subplots

fig, ax = plt.subplots()
   
# Define Data Coordinates

x = np.linspace(0, 20 , 100)
y = np.sin(x)

# Plot

plt.plot(x, y)

# Set ticks

ax.set_xticks(np.arange(0, len(x)+1, 5))

# Add Title
 
fig.suptitle('set_xticks Example', fontweight ="bold")
               
# Display
               
plt.show()
