brojevi = []

while True:
    unos = input("Unesite broj ili 'Done' za kraj: ")
    if unos.lower() == 'done':
        break
    
    try:
        broj = float(unos)
        brojevi.append(broj)
    except:
        print("Pogresan unos. Molimo unesite broj koji je validan.")


print(f"Broj unesenih brojeva: {len(brojevi)}")
print(f"Srednja vrijednost: {sum(brojevi) / len(brojevi):.2f}")
print(f"Minimalna vrijednost: {min(brojevi)}")
print(f"Maksimalna vrijednost: {max(brojevi)}")
    
brojevi.sort()
print("Sortirana lista:", brojevi)


