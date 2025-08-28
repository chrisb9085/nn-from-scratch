import math

softmax_output = [0.7, 0.1, 0.2] # Each value represents the probability of each class

target_class = 0 # At index 0 it is "hot"

target_output = [1, 0, 0] # One-hot encoded target output

loss = -(math.log(softmax_output[0])*target_output[0] +
         math.log(softmax_output[1])*target_output[1] +
         math.log(softmax_output[2])*target_output[2])

print("Loss:", loss)
print(-math.log(0.7)) # Confidence is higher, so loss is lower
print(-math.log(0.5)) # Confidence is lower, so loss is higher
