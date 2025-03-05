def total_euro(sati, satnica):
    return sati * satnica

radni_sati = float(input("Radni sati: "))
eura_po_satu = float(input("eura/h: "))

zarada = total_euro(radni_sati, eura_po_satu)
print(f"Ukupno: {zarada:.2f} eura")


