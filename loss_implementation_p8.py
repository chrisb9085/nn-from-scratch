import numpy as np

softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])

class_targets = [0, 1, 1]

print(softmax_outputs[[0, 1, 2], class_targets])
# Since softmax_outputs is a 2D array, we need to provide the row indices as well

# In Java, this is equivalent to:
# softmaxOutputs[0][classTargets[0]], softmaxOutputs[1][classTargets[1]], softmaxOutputs[2][classTargets[1]]

# Wrap it in a log function to get the loss
loss = -np.log(softmax_outputs[range(len(softmax_outputs)), class_targets])
print(loss)

# However, this still proposes a problem if we have np.log of 0
# We can clip the values to a minimum (insignificat) value, such as 1e-7
# This is called numerical stability
y_pred = 0 # For example so code runs
y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
