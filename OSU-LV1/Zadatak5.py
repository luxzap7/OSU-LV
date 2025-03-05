ham_rijeci = []
spam_rijeci = []
spam_s_usklicnikom = 0

with open('SMSSpamCollection.txt', 'r', encoding='utf-8') as file:
    for linija in file:
        vrsta, poruka = linija.split('\t', 1)
        rijeci = poruka.split()
        
        if vrsta == 'ham':
            ham_rijeci.append(len(rijeci))
        elif vrsta == 'spam':
            spam_rijeci.append(len(rijeci))
            if poruka.strip().endswith('!'):
                spam_s_usklicnikom += 1

print(f"Prosjecan broj rijeci u ham porukama: {sum(ham_rijeci) / len(ham_rijeci):.2f}")
print(f"Prosjecan broj rijeci u spam porukama: {sum(spam_rijeci) / len(spam_rijeci):.2f}")
print(f"Broj spam poruka koje zavrsavaju usklicnikom: {spam_s_usklicnikom}")
