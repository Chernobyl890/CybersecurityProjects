# calculate the entropy for a dice roll
from math import log2
import numpy as np
from numpy import maximum
from scipy.stats import entropy
# the number of events

# probability of one event
# calculate entropy
entropy = -(0.41/0.8)*log2(0.41/0.8) - (0.39/0.8)*log2(0.39/0.8)
# print the result
print("entropy: " , entropy)

#base = 2
#pk = np.array([8/20, 4/20])
#H = entropy(pk, base=base)
#print("entropy: ", H)

# Calculate Max
#max = 1 - maximum(7/10, 3/10)
# print the result
#print("max: ", max)