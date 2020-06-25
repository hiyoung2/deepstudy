first = input()
second = input()

first = int(first)

second_3 = int(second[-1])
second_2 = int(second[-2])
second_1 = int(second[-3])

third = first*second_3
fourth = first*second_2
fifth =first*second_1

sum = (third+(fourth*10)+(fifth*10**2))

sixth = sum

print(third)
print(fourth)
print(fifth)
print(sixth)