import os

name_path = '/home/huangjiu/projects/DCFM_edge_SA_GCM_1/2eval/pred2'
for root, dirs, files in os.walk(name_path):
    for dir in dirs:
        os.rename(os.path.join(name_path,dir),os.path.join(name_path,dir.split("_Smeasure")[0]))


final_name = ""
name_list = os.listdir(name_path)
for name in name_list:
    final_name = final_name + name + "+"

print(final_name)
