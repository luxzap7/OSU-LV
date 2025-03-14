import numpy as np
import matplotlib.pyplot as plt

white = np.zeros((50,50),dtype=int)
black = np.ones((50,50),dtype=int)

up = np.hstack((black,white))
down = np.hstack((white,black))

img = np.vstack((up,down))

plt.imshow(img,cmap="gray")
plt.axis([0,100,0,100])
plt.show()