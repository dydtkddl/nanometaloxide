import os 
allcount = 962399
dir = "./repos/OXIDE/prediction_log_%s.txt"%allcount

with open(dir, "r") as f:
  data = f.read()

data[:100]  
import ast 
import datetime 
each_points = data.strip().split("\n\n")
print(len(each_points))
print(each_points[0])
c1 = 0
c2 = 0
both = 0
titles = []
onlyna = []
onlymain = []
from tqdm import tqdm
el = 0
for points in tqdm(each_points):
  doi_start = points.find("DOI: ")
  pub_start = points.find("Publication Date: ")
  na_start = points.find("Is Nanoparticle: ")
  ma_start = points.find("Main Subject: ")
  
  try:
    text_ = points[:doi_start]
    doi = points[doi_start+len("DOI: ") : pub_start].strip()
    pubdate =points[pub_start+len("Publication Date: ") : na_start].strip()
    is_na = ast.literal_eval( points[na_start+len("Is Nanoparticle: ") : ma_start].strip())
    mainsub = ast.literal_eval( points[ma_start+len("Main Subject: ") : ].strip())
    # if (mainsub != [] and is_na !=[]):
    #   print(mainsub, is_na)
    if (is_na and mainsub):
      both+=1
      titles.append((text_, is_na, mainsub,pubdate,doi))
    elif (is_na):
      onlyna.append((text_, is_na, mainsub))
    elif (mainsub):
      onlymain.append((text_,mainsub))
    else:
      el +=1
  except:
    pass
import json
from collections import defaultdict
import re

dic = {}

def extract_year_from_date(date_str):
    # 정규 표현식을 사용하여 연도 추출
    year_match = re.match(r'^\d{4}', date_str)
    if year_match:
        return int(year_match.group())
    return None

def count_materials_per_year(data):
    for da in data:
        fir = da[1]
        sec = da[2]
        year = extract_year_from_date(da[3])
        for f in fir:
            if f in sec:
                if year not in dic.keys():
                    dic[year] = {f: 1}
                else:
                    if f in dic[year].keys():
                        dic[year][f] += 1
                    else:
                        dic[year][f] = 1
    return dic

# # 함수 호출 및 결과 출력
# year_material_count = count_materials_per_year(titles)

# # 결과 출력
# for year, materials in year_material_count.items():
#     print(f"Year: {year}")
#     for material, count in materials.items():
#         print(f"  Material: {material}, Count: {count}")
# 함수 호출 및 결과 출력
year_material_count = count_materials_per_year(titles)
# tz
# 로그 파일 경로 설정
log_file_path = "./repos/OXIDE/yearly_material_count_log%s.txt"%allcount

# 로그 파일 생성 및 결과 저장
with open(log_file_path, "w") as log_file:
    for year, materials in year_material_count.items():
        log_file.write(f"Year: {year}\n")
        for material, count in materials.items():
            log_file.write(f"  Material: {material}, Count: {count}\n")

print(f"로그 파일이 '{log_file_path}'에 생성되었습니다.")