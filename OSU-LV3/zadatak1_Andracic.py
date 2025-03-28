import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Učitavanje podataka

df = pd.read_csv('data_CO2_emission.csv')




# a) 

print(f"Broj mjerenja: {len(df)}")

print(df.dtypes)

print(df.isnull().sum())

print(f"Broj dupliciranih redova: {df.duplicated().sum()}")

df = df.dropna().drop_duplicates()

categorical_columns = ['Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type']
df[categorical_columns] = df[categorical_columns].astype('category')


# b)

df_sorted = df.sort_values('Fuel Consumption City (L/100km)')

print("Tri automobila s najmanjom gradskom potrošnjom:")
print(df_sorted[['Make', 'Model', 'Fuel Consumption City (L/100km)']].head(3))

print("\nTri automobila s najvećom gradskom potrošnjom:")
print(df_sorted[['Make', 'Model', 'Fuel Consumption City (L/100km)']].tail(3))



# c)

filtered_df = df[(df['Engine Size (L)'] >= 2.5) & (df['Engine Size (L)'] <= 3.5)]
print(f"Broj vozila s veličinom motora između 2.5 i 3.5 L: {len(filtered_df)}")
print(f"Prosječna CO2 emisija za ova vozila: {filtered_df['CO2 Emissions (g/km)'].mean():.2f} g/km")


# d)

audi_df = df[df['Make'] == 'Audi']
print(f"Broj mjerenja za vozila proizvođača Audi: {len(audi_df)}")

audi_4cyl = audi_df[audi_df['Cylinders'] == 4]
print(f"Prosječna emisija CO2 za Audi vozila s 4 cilindra: {audi_4cyl['CO2 Emissions (g/km)'].mean():.2f} g/km")

# e)

grouped_cylinder_cars = df.groupby('Cylinders')

cylinder_counts = grouped_cylinder_cars.size()
print(f"Broj vozila po broju cilindara: \n{cylinder_counts}")
average_co2_per_cylinder_amount = grouped_cylinder_cars['CO2 Emissions (g/km)'].mean()
print(f"Prosječna emisija CO2 po broju cilindara filtriranih auta: \n{average_co2_per_cylinder_amount}")

plt.figure()

average_co2_per_cylinder_amount.plot(kind='bar')

plt.title('Prosječna emisija CO2 po broju cilindara') 
plt.xlabel('Broj cilindara')
plt.ylabel('Prosječna emisija CO2 (g/km)')
plt.show()


# f)

fuel_consumption = df.groupby('Fuel Type')['Fuel Consumption City (L/100km)']
print("Prosječna gradska potrošnja:")
print(fuel_consumption.mean())
print("\nMedijalna gradska potrošnja:")
print(fuel_consumption.median())

# g)

diesel_4cyl = df[(df['Cylinders'] == 4) & (df['Fuel Type'] == 'D')]
max_consumption = diesel_4cyl.loc[diesel_4cyl['Fuel Consumption City (L/100km)'].idxmax()]
print(f"Vozilo: {max_consumption['Make']} {max_consumption['Model']}")
print(f"Gradska potrošnja: {max_consumption['Fuel Consumption City (L/100km)']} L/100km")

# h)

manual_transmission = df[df['Transmission'].str.startswith('M')]
print(f"Broj vozila s ručnim mjenjačem: {len(manual_transmission)}")

# i)

numeric_df = df.select_dtypes(include=[np.number])
correlation = numeric_df.corr()
print(correlation)

# Ukazuje se korelacija izmedu velicine motora / kolicine CO2 / broja cilindra te potrosnje goriva 