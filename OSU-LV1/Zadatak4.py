from collections import Counter

with open('song.txt', 'r') as file:
    tekst = file.read().lower()
    rijeci = tekst.split()
    
brojac = Counter(rijeci)

print(f"Broj rijeci koje se pojavljuju samo jednom: {sum(1 for count in brojac.values() if count == 1)}")
print("Rijeci koje se pojavljuju samo jednom:")
for rijec, broj in brojac.items():
    if broj == 1:
        print(rijec)

