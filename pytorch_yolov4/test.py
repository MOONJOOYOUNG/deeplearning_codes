import os


filename = './data/dataset/train/train/apple_22.jpg'
filename = os.path.splitext(os.path.basename(filename))[0]
print(filename)
lv, no = filename.split('_')
print(lv)
print(no)
# print(filename)
# lv, no = filename.split("_")
# print(lv, no)
# lv = lv.replace("level_", "")
# print(lv,no)
# print(int(lv+no))
# # lv, no = os.path.splitext
# # print(lv,no)
# lv = lv.replace("level_", "")
# print(lv,no)