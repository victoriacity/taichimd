import taichi as ti
import numpy as np

'''
Do not change! The code was not tested 
for dimensions other than 3
'''
DIM =3

PI = 3.14159265358979

#IDENTITY = ti.Matrix(np.eye(DIM, dtype=np.float))

COLOR_MOLECULES = [[0.02, 0.85, 0.86],
              [0.8, 0.8, 0.8],
              [0.9, 0, 0],#[0.5, 0.5, 0.5],
              [1, 0, 0],
              [0, 1, 0],
              [0, 0, 1],
            ]