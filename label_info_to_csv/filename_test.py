import glob


def file_path_save():
    file_list= sorted(glob.glob("D:/test/*.jpg"))

    print(file_list)

    filename = []

    cnt = 0
    for i in range(len(file_list)):
        filename_jpg = file_list[i].split("\\")[1]
        filename_only = filename_jpg[:-4]
        print(filename_only)

        filename.append(filename_only)
    
    # print(filename)

    for j in range(len(filename)):
        f = open("D:/temp/filelist.txt", 'a')
        f.write(filename[j] + "\n")


if __name__ == '__main__':file_path_save()

