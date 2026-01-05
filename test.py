import csv
import pandas as pd
import os

my_scores = {'可愛': 3, '漂亮': 3, '有趣': 3, '氣質': 3, '知性': 3}
categories = ["可愛", "漂亮", "有趣", "氣質", "知性"]


# # 開啟 CSV 檔案並讀取內容

# best_score = 100
# best_coffee = None

# with open('coffee_item.csv', mode='r', encoding='utf-8') as file:
#     reader = csv.DictReader(file)

#     for row in reader:
#         diff_score = 0
#         coffee_name = row['coffee_name']
#         for cat in categories:
#             diff_score += abs(float(my_scores[cat])-float(row[cat]))

#         if diff_score < best_score:
#             best_score = diff_score
#             best_coffee = coffee_name

#         print(coffee_name,diff_score)

# print("best", best_coffee, best_score)

# filepath = "coffee_info"

# if os.path.exists(os.path.join(filepath,"絳絳.webp")):
#     coffee_image_path = os.path.join(filepath,"絳絳.webp")
# elif os.path.exists(os.path.join(filepath,"絳絳.png")):
#     coffee_image_path = os.path.join(filepath,"絳絳.png")

# print(coffee_image_path)
# print("111",os.path.exists(coffee_image_path))

print(list(my_scores.keys()))
print(type(list(my_scores.keys())))