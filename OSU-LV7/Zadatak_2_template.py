import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans

# ucitaj sliku
img = mpimg.imread("imgs\\test_1.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()

# Broj različitih boja u slici
unique_colors = np.unique(img_array, axis=0)
print(f"Broj različitih boja u slici: {len(unique_colors)}")

# Primjena KMeans algoritma
k = 5  # Postavite broj grupa (K)
kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(img_array)
labels = kmeans.predict(img_array)

# Zamjena vrijednosti elemenata s pripadajućim centrom
centers = kmeans.cluster_centers_
img_array_aprox = centers[labels]
img_aprox = np.reshape(img_array_aprox, (w, h, d))

# Prikaz rezultantne slike
plt.figure()
plt.title(f"Rezultantna slika (K={k})")
plt.imshow(img_aprox)
plt.tight_layout()
plt.show()

# Grafički prikaz ovisnosti J o broju grupa K
inertias = []
k_values = range(1, 11)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(img_array)
    inertias.append(kmeans.inertia_)

plt.figure()
plt.plot(k_values, inertias, marker='o')
plt.title("Ovisnost J o broju grupa K")
plt.xlabel("Broj grupa K")
plt.ylabel("Inercija (J)")
plt.tight_layout()
plt.show()

# Prikaz elemenata slike koji pripadaju jednoj grupi
for i in range(k):
    binary_img = (labels == i).astype(float)
    binary_img = np.reshape(binary_img, (w, h))
    plt.figure()
    plt.title(f"Grupa {i+1}")
    plt.imshow(binary_img, cmap='gray')
    plt.tight_layout()
    plt.show()
