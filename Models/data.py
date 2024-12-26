import pandas as pd
import matplotlib.pyplot as plt
# Specify the delimiter as '|'
# df = pd.read_csv(r'archive (1)\flickr30k_images\results.csv', delimiter='|')
# print(df.head())  # This will print the first five rows of the dataframe to verify it's reading correctly
# plt.imshow(plt.imread(r'archive (1)\flickr30k_images\flickr30k_images\\'+  df['image_name'][0]))
# plt.show()
# def new_path(path):
#     path=r'archive (1)\flickr30k_images\flickr30k_images\\'+path
#     return path
# df['image_name']=df['image_name'].apply(new_path)
# df.to_csv('Images.csv')
df=pd.read_csv(r'archive (1)\flickr30k_images\Images.csv',delimiter='|')
print(df.head())