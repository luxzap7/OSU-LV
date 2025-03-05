x = 23
print ( x )
x = x + 7
print ( x )

x = 23
y = x > 10
print ( y )



i = 5
while i > 0:
    print ( i )
    i = i - 1
print (" Petlja gotova ")

for i in range (0 , 5 ):
    print ( i )


fruit = 'banana'
index = 0
count = 0

while index < len(fruit):
    letter = fruit[index]
    if letter == 'a':
        count = count + 1

    print(letter)
    index = index + 1

print(count)

print(fruit[0:3])
print(fruit[0:])
print(fruit[2:6:1])
print(fruit[0:-1])
