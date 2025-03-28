import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Učitavanje podataka
df = pd.read_csv("data_C02_emission.csv")




# a)

plt.figure(figsize=(10, 6))
df['CO2 Emissions (g/km)'].hist(bins=30)
plt.title('Histogram emisije CO2')
plt.xlabel('Emisija CO2 (g/km)')
plt.ylabel('Broj vozila')
plt.show()

# Histogram prikazuje distribuciju emisije CO2 gdje je vecina vozila u nizem ili srednjem rasponu emisije

# b)


plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['Fuel Consumption City (L/100km)'], df['CO2 Emissions (g/km)'],
c=df['Fuel Type'].astype('category').cat.codes, cmap='viridis')
plt.colorbar(scatter, label='Tip goriva')
plt.title('Odnos gradske potrošnje i emisije CO2')
plt.xlabel('Gradska potrošnja (L/100km)')
plt.ylabel('Emisija CO2 (g/km)')
plt.show()
# Prikazana je snazna korelacija izmedu gradske potrosnje i emisije CO2

# c)

plt.figure(figsize=(10, 6))
df.boxplot(column='Fuel Consumption Hwy (L/100km)', by='Fuel Type')
plt.title('Razdioba izvangradske potrošnje po tipu goriva')
plt.suptitle('') # Uklanjanje automatskog naslova
plt.ylabel('Izvangradska potrošnja (L/100km)')
plt.show()


# da primjecuje se

# d)

fuel_counts = df.groupby('Fuel Type').size()
plt.figure(figsize=(10, 6))
fuel_counts.plot(kind='bar')
plt.title('Broj vozila po tipu goriva')
plt.xlabel('Tip goriva')
plt.ylabel('Broj vozila')
plt.xticks(rotation=45)
plt.show()

# e)

avg_co2 = df.groupby('Cylinders')['CO2 Emissions (g/km)'].mean()
plt.figure(figsize=(10, 6))
avg_co2.plot(kind='bar')
plt.title('Prosječna emisija CO2 po broju cilindara')
plt.xlabel('Broj cilindara')
plt.ylabel('Prosječna emisija CO2 (g/km)')
plt.show()
