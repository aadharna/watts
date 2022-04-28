import os

path = "source/"
for filename in os.listdir(path):
    if "watts." in filename:
        output_lines = []
        with open(path + filename, 'r') as f:
            for line in f.readlines():
                if ".. automodule:: watts." in line:
                    output_lines.append(f'.. automodule:: {line.split("watts.")[1]}')
                else:
                    output_lines.append(line)
        with open(path + filename, 'w') as f:
            f.writelines(output_lines)
