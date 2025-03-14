import numpy as np
import matplotlib.pyplot as plt

x=[1.0,2.0,3.0,3.0,1.0]
y=[1.0,2.0,2.0,1.0,1.0]

plt.plot (x , y , 'b', linewidth = 5 , marker =".", markersize =5 )
plt.axis ([0 ,4 , 0 , 4])

plt.xlabel ('x os')
plt.ylabel ('y os')
plt.title ('Četverokut')

plt.fill(x,y,color="blue",alpha=0.3)
plt.show()
