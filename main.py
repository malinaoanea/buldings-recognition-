import os
from os import listdir
import shutil

cats_files = listdir("data/train/cats")
cats_files_val = listdir("data/validation/cats")

print(len(cats_files), len(cats_files_val))

cats_files = listdir("data/train/doggos")
cats_files_val = listdir("data/validation/doggos")
print(len(cats_files), len(cats_files_val))
# # print(len(files))
# nr_cat = 0
# nr_doggos = 0
# for f in range(2000):

#     original = "data/train/cats/" + files[f]
#     target = "data/validation/cats/" + files[f]

#     shutil.move(original, target)

# files = listdir("data/train/doggos")
# # print(len(files))
# nr_cat = 0
# nr_doggos = 0
# for f in range(2000):

#     original = "data/train/doggos/" + files[f]
#     target = "data/validation/doggos/" + files[f]

#     shutil.move(original, target)


    

# print("nr pisici", nr_cat)
# print("nr_doggos", nr_doggos)

