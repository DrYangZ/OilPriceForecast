import pdfplumber
import re
import pandas as pd

excel_path = 'Registration form_all_V03.xlsm'
KUKA_path = 'RS26 Certificate 202407.pdf'
PLC_path = 'PLC Certificate 202407.pdf'
PLC_badge_path = 'PLC06 badge 202407.pdf'
KUKA_badge_path = 'RS26 badge 202407.pdf'
df = pd.read_excel(excel_path, sheet_name='Allg.', engine='openpyxl')
rows_to_check = df.iloc[801:861]
passed_rows = rows_to_check[rows_to_check['Unnamed: 9'] == 'PASS']
# print(rows_to_check['Unnamed: 9'] == 'PASS')
# print(rows_to_check)
# print(passed_rows)
names = passed_rows['Unnamed: 1'].tolist()
ids = passed_rows['Unnamed: 3'].tolist()
companies = passed_rows['Unnamed: 5'].tolist()
# print(rows_to_check['Unnamed: 1'])
# print(type(ids))
# print(rows_to_check.iterrows())

# 定义一个正则表达式模式来匹配人员信息
certificate_pattern = re.compile(r'([\u4e00-\u9fa5]+)\s+(\d{18})\s+([\u4e00-\u9fa5]+公司)')
badge_pattern = re.compile(r'(?P<name>[\u4e00-\u9fa5]+)\s+ID:\s+(?P<id>\d{18})')

wrong_list = []
# 打开PDF文件
with pdfplumber.open(PLC_badge_path) as pdf:
    # 遍历每一页
    num = 0
    wrong_num = 0
    for page in pdf.pages:
        # 提取当前页的文本
        text = page.extract_text()
        if text:
            # 使用正则表达式查找所有匹配项
            matches = badge_pattern.findall(text)
            print(matches)
            for match in matches:
                num += 1
                name, id_num = match
                # name, id_num, company = match
                if name in names:
                    index = names.index(name)
                    if id_num == ids[index]:
                        # if id_num == ids[index] and company == companies[index]:
                        print(f'考生：{name} 的信息匹配正确！')
                    else:
                        wrong_num += 1
                        wrong_list.append(name)
                        print(f'考生：{name} 的信息匹配错误！')
                else:
                    wrong_num += 1
                    wrong_list.append(name)
                    print(f'考生：{name}未通过考试却出现在了证书名单中！')
    print(f'该pdf页面中共有{num}个人员信息！')
    print(f'有{wrong_num}人信息错误\n，错误名单为：{wrong_list}')