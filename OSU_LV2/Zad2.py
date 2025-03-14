import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('data.csv',delimiter=',', skip_header=1)

print(f"Mjerenja izvršena nad {np.shape(data)[0]} osoba")


x = data[:,1]
y = data[:,2]
plt.scatter(x,y,color="red")
plt.xlabel("Visina osobe")
plt.ylabel("Tezina osobe")
plt.title("Prikaz omjera mase i visine osoba")
plt.show()

x = data[::50,1]
y = data[::50,2]
plt.scatter(x,y,color="red")
plt.xlabel("Visina osobe")
plt.ylabel("Tezina osobe")
plt.title("Prikaz omjera mase"
" i visine osoba")
plt.show()


print(f"Minimalna visina: {np.min(data[:,1])}\n Maximalna visina: {np.max(data[:,1])}\n Average visina: {np.mean(data[:,1])}")


zene = data[data[:,0] == 0]
muski = data[data[:,0] == 1]
print(f"Minimalna visina zena: {np.min(zene[:,1])}\n Maximalna visina zena: {np.max(zene[:,1])}\n Average visina zena: {np.mean(zene[:,1])}")
print(f"Minimalna visina muskaraca: {np.min(muski[:,1])}\n Maximalna visina muskaraca: {np.max(muski[:,1])}\n Average visina muskaraca: {np.mean(muski[:,1])}")
