import pandas as pd

import os

# images = os.listdir("C:\work\web scrape\cropped_images")
old_csv = pd.read_csv("C:\work\web scrape\data.csv")
print(old_csv.head())
new_csv_data = []
for index, row in old_csv.iterrows():
    print(row)
    image_name =  row["image"]
    print(f"image name is {image_name}")
    latex = row["latex"]
    print(f"latex is {latex}")
    latex = "$"+latex+"$"
    new_csv_data.append((image_name,latex))

# for image in images:
    # get the corresponding label
    # try:
        # latex = old_csv.loc[old_csv['image']==image]
        # latex = str(latex['latex'].item())
        # print(f"latex recieved is {latex}")
        # new_csv_data.append((image,latex))
    # except:
        # print("no corresponding latex found")
        # break

df = pd.DataFrame(new_csv_data)
df.to_csv('$_data.csv') 
    