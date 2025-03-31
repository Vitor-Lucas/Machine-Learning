
lines = []
with open('/home/vitor/Coding/Python/Machine-Learning/Datasets/iris/Iris.csv', 'r') as file:
    for line in file.readlines():
        # print(line)
        lines.append(line)

species = []
for line in lines[1:]:
    specie = line.split(',')[-1]
    species.append(specie.replace('\n', ''))
species = {specie: f'{i}' for i,specie in enumerate(set(species))}

# print(species)

new_lines = []
for line in lines[1:]:
    *values, specie = line.split(',')
    new_lines.append(','.join([*values, species[specie.replace('\n', '')]]))

# print(new_lines)

with open('/home/vitor/Coding/Python/Machine-Learning/Datasets/iris/Iris_processed.csv', 'w') as file:
    for line in new_lines:
        file.write(line + "\n")