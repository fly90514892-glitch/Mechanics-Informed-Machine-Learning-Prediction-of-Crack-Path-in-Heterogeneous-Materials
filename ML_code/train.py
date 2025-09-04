#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import time
from pathlib import Path

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, jaccard_score)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Flatten, Dropout,
                                     LayerNormalization, MultiHeadAttention)

#
# 1. I/O & data preparation
#
DATA_DIR        = Path('../512')          # << directory with *_map*.txt
input_time_steps = 0
total_time_steps = 1
time_steps_interval = 1

initialCase = 1
finalCase   = 631 * 2

matrix_rows, matrix_cols = 9, 4           # every map is 9 x 4 = 36 pixels
feat_dim = matrix_rows * matrix_cols

print(' Reading text maps ')
matrix_sequence = []
for case in range(initialCase, finalCase + 1):
    case_mats = []
    for i in range(input_time_steps, total_time_steps + 1, time_steps_interval):
        file_path = DATA_DIR / f"{case}_map{i}.txt"
        with open(file_path, 'r') as f:
            lines  = [l.strip() for l in f]
            matrix = [list(map(float, ln.split(','))) for ln in lines]
            case_mats.append(matrix)
    matrix_sequence.append(case_mats)

matrix_sequence = np.array(matrix_sequence, dtype=np.float32)
# shape  (num_cases, time_steps=2, 9, 4)

# Input = t  ;  Target = t
X = matrix_sequence[:, :-1]     # (N,1,9,4)
y = matrix_sequence[:, 1:]      # (N,1,9,4)

# reshape to (N, 1, 36) and (N, 36)
X = X.reshape((X.shape[0], -1, feat_dim))   # (N,1,36)
y = y.reshape((y.shape[0], -1, feat_dim))   # (N,1,36)
y = y[:, 0, :]                              # squeeze  (N,36)

print('X  shape:', X.shape)
print('y  shape:', y.shape)

#
# 2. Model definition
#
def transformer_model(input_shape, output_dim):
    inp = Input(shape=input_shape)
    x   = MultiHeadAttention(num_heads=8, key_dim=32)(inp, inp)
    x   = LayerNormalization(epsilon=1e-6)(x)
    x   = Flatten()(x)
    x   = Dense(512, activation='relu')(x)
    x   = Dropout(0.1)(x)
    out = Dense(output_dim, activation='linear')(x)
    return Model(inp, out)

#
# 3. Cross-validation
#
print(' Performing 5-fold cross-validation ')
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_metrics = []

t0 = time.time()
for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    print(f'Fold {fold}/5')
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]

    model_fold = transformer_model(input_shape=X.shape[1:], output_dim=feat_dim)
    model_fold.compile(loss='mse', optimizer='adam')

    model_fold.fit(X_train_fold, y_train_fold,
                   epochs=500, batch_size=1, verbose=1,
                   validation_data=(X_val_fold, y_val_fold))

    y_pred_val = model_fold.predict(X_val_fold, verbose=0)
    y_pred_bin = (y_pred_val >= 0.5).astype(int).ravel()
    y_true_bin = (y_val_fold >= 0.5).astype(int).ravel()

    acc  = accuracy_score(y_true_bin, y_pred_bin)
    prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    rec  = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1   = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    iou  = jaccard_score(y_true_bin, y_pred_bin, zero_division=0)

    fold_metrics.append([acc, prec, rec, f1, iou])
    print(f'Fold {fold} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, '
          f'Recall: {rec:.4f}, F1: {f1:.4f}, IoU: {iou:.4f}')

fold_metrics = np.array(fold_metrics)
mean_metrics = fold_metrics.mean(axis=0)
std_metrics = fold_metrics.std(axis=0)

print('\nCross-validation results (mean  std):')
print(f'Accuracy : {mean_metrics[0]:.4f}  {std_metrics[0]:.4f}')
print(f'Precision: {mean_metrics[1]:.4f}  {std_metrics[1]:.4f}')
print(f'Recall   : {mean_metrics[2]:.4f}  {std_metrics[2]:.4f}')
print(f'F1 score : {mean_metrics[3]:.4f}  {std_metrics[3]:.4f}')
print(f'IoU      : {mean_metrics[4]:.4f}  {std_metrics[4]:.4f}')
print(f'Cross-validation finished in {time.time()-t0:.1f} s')

cross_val_data = np.column_stack((mean_metrics, std_metrics))
np.savetxt('cross_val_metrics.txt',
           cross_val_data,
           header='Accuracy,Precision,Recall,F1,IoU,Acc_std,Prec_std,Rec_std,F1_std,IoU_std',
           fmt='%.6f', delimiter=',')
print('cross_val_metrics.txt written')

print('\nPerforming single-split training ')
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, random_state=42, shuffle=True)

model = transformer_model(input_shape=X.shape[1:], output_dim=feat_dim)
model.compile(loss='mse', optimizer='adam')

t0 = time.time()
hist = model.fit(X_train, y_train,
                 epochs=500, batch_size=1, verbose=1,
                 validation_data=(X_val, y_val))
print(f'Single-split training finished in {time.time()-t0:.1f} s')

print('Computing single-split validation metrics ')
y_pred_val = model.predict(X_val, verbose=0)

y_pred_bin = (y_pred_val >= 0.5).astype(int).ravel()
y_true_bin = (y_val       >= 0.5).astype(int).ravel()

acc  = accuracy_score(y_true_bin, y_pred_bin)
prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
rec  = recall_score   (y_true_bin, y_pred_bin, zero_division=0)
f1   = f1_score       (y_true_bin, y_pred_bin, zero_division=0)
iou  = jaccard_score  (y_true_bin, y_pred_bin, zero_division=0)

print(f'Accuracy : {acc :.4f}')
print(f'Precision: {prec:.4f}')
print(f'Recall   : {rec :.4f}')
print(f'F1 score : {f1 :.4f}')
print(f'IoU      : {iou:.4f}')

np.savetxt('metrics.txt',
           np.array([[acc, prec, rec, f1, iou]]),
           header='Accuracy,Precision,Recall,F1,IoU',
           fmt='%.6f', delimiter=',')
print('metrics.txt written')

model.save('transformer_model', save_format='tf')
print('Model saved to ./transformer_model')

loss_hist = np.column_stack((np.array(range(1, len(hist.history['loss'])+1)),
                             hist.history['loss'],
                             hist.history['val_loss']))
np.savetxt('loss.txt', loss_hist, delimiter=',',
           header='Epoch,Training Loss,Validation Loss', fmt='%.6f')
print('loss.txt written  script finished.')