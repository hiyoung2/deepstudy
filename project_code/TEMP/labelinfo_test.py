import os

label_list = "D:/test/"
outfile_name = "label_info.txt"

out_file = open(outfile_name, "w")

files = os.listdir(label_list)

for filename in files:
    if ".txt" not in filename :
        continue
    file = open(label_list + filename)
    for line in file:
        out_file.write(line)
    out_file.write("")
    file.close()
out_file.close()


