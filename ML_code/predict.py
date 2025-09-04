#!/usr/bin/env python
# coding: utf-8

# In[32]:


import numpy as np
import time
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, LayerNormalization, MultiHeadAttention

input_time_steps = 0
total_time_steps = 1
time_steps_interval = 1

initialCase = 1
finalCase = 1

matrix_rows = 9  # Number of rows in each matrix
matrix_cols = 4  # Number of columns in each matrix

############################ Predict ############################
from tensorflow.keras.models import load_model

model = load_model('transformer_model')
print("Model loaded successfully.")

# Define the matrix sequence
for case in range(initialCase, finalCase + 1):
    # Define the matrix sequence
    input_matrix_sequence = []
    file_path = f"process_zone{input_time_steps}.txt"
    with open(file_path, 'r') as file:
        lines = file.readlines()
        input_matrix = [list(map(float, line.strip().split(','))) for line in lines]
        input_matrix_sequence.append(input_matrix)
    input_matrix_sequence = np.array(input_matrix_sequence)
    print(input_matrix_sequence.shape)

    # Generate predictions for multiple time steps
    for timestep in range(input_time_steps + time_steps_interval, total_time_steps + 1, time_steps_interval):
        # Reshape the input matrix
        input_matrix_sequence_reshaped = input_matrix_sequence.reshape((input_matrix_sequence.shape[0], input_matrix_sequence.shape[0], input_matrix_sequence.shape[1] * input_matrix_sequence.shape[2]))
        print(input_matrix_sequence_reshaped.shape)
            
        # Make a prediction for the next matrix
        prediction = model.predict(input_matrix_sequence_reshaped)
        print(prediction.shape)

        # Reshape the predicted matrix
        predicted_matrix = prediction.reshape((matrix_rows, matrix_cols))
        print(predicted_matrix.shape)
        
        # Round the values to binary (assuming it's a binary classification task)
        predicted_matrix = np.round(predicted_matrix)

        # Save the predicted matrix
        output_file = f"process_zone_predict{timestep}.txt"
        np.savetxt(output_file, predicted_matrix, delimiter=',', fmt='%.5f')
        print(f"Predicted Matrix saved in {output_file}")

        # Update the input matrix for the next time step
        #input_matrix_sequence = np.concatenate([input_matrix_sequence, predicted_matrix.reshape((1, predicted_matrix.shape[0], predicted_matrix.shape[1]))], axis=1)
        input_matrix_sequence = predicted_matrix.reshape((1, predicted_matrix.shape[0], predicted_matrix.shape[1]))
        print(input_matrix_sequence.shape)

print("Prediction completed for all time steps.")


# In[ ]:




