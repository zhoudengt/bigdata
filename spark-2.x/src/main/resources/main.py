import time
from DrissionPage import SessionPage
from bs4 import BeautifulSoup
from re import findall
from tqdm import tqdm
from hashlib import md5
import random
import json
import locale
from datetime import datetime, timedelta
def get(url):
    data={}
    page.get(url)
    name=findall('daily/(.*?).html',url)[0]
    data['星座']=no_chian(name)
    data['每日运势']=no_chian(page.ele('#content').text)
    data['每日饮食'] = no_chian(page.ele('#content-food').text)
    data['每日主页运势'] = no_chian(page.ele('#content-home').text)
    data['每日狗运势'] = no_chian(page.ele('#content-dog').text)
    data['每日青少年运势'] = no_chian(page.ele('#content-teen').text)
    data['每日猫运势'] = no_chian(page.ele('#content-cat').text)
    data['每日奖金运势'] = no_chian(page.ele('#content-bonus').text)
    for i in page.eles('x://*[@rel="noopener nofollow"]'):
        a = findall("""updateContent\(.*?\)""", i.html)[0]
        if get_tomorrow_date() in a:
            a = findall("""this, '(.*?)', '.*?', 'overview', '(.*?)'""", i.html)
            urls = f'https://www.astrology.com/horoscope/daily/overview/{name}/{a[0][0]}/{a[0][1]}/'
            page.get(urls)
            data['每日运势'] = no_chian(BeautifulSoup(page.json['horoscope'], 'lxml').get_text())
    data.update(get2(name))
    return data
def get2(url):
    data={}
    page.get(f'https://www.astrology.com/horoscope/weekly-overview/{url}.html')
    a1=[i.text for i in BeautifulSoup(page.html,'lxml').findAll('p')]
    a1.pop(0)
    a1.pop(0)
    data['每周运势']=nota(str('\n'.join(a1)).split('To read more of your weekly horoscope, subscribe to Astrology+.')[0])
    page.get(f'https://www.astrology.com/horoscope/monthly-overview/{url}.html')
    a1 = [i.text for i in BeautifulSoup(page.html, 'lxml').findAll('p')]
    a1.pop(0)
    a1.pop(0)
    data['每月运势'] = nota(str('\n'.join(a1)).split('Find outwhat lies ahead')[0])
    page.get(f'https://www.astrology.com/horoscope/yearly-love/{url}.html')
    a1 = [i.text for i in BeautifulSoup(page.html, 'lxml').findAll('p')]
    a1.pop(0)
    a1.pop(0)
    data['每年运势'] = nota(str(str('\n'.join(a1).split('Are you and your love interest meant to be?')[0])).split('Find outwhat lies ahead')[0])
    page.get(f'https://www.astrology.com/horoscope/daily-love/{url}.html')
    for i in page.eles('x://*[@rel="noopener nofollow"]'):
        a = findall("""updateContent\(.*?\)""", i.html)[0]
        if get_tomorrow_date() in a:
            a = findall("""this, '(.*?)', '.*?', 'love', '(.*?)'""", i.html)
            urls = f'https://www.astrology.com/horoscope/daily/love/{url}/{a[0][0]}/{a[0][1]}/'
            page.get(urls)

            data['爱情运势'] = no_chian(BeautifulSoup(page.json['horoscope'], 'lxml').get_text())
    page.get(f'https://www.astrology.com/horoscope/daily-work/{url}.html')
    for i in page.eles('x://*[@rel="noopener nofollow"]'):
        a = findall("""updateContent\(.*?\)""", i.html)[0]
        if get_tomorrow_date() in a:
            a = findall("""this, '(.*?)', '.*?', 'work', '(.*?)'""", i.html)
            urls = f'https://www.astrology.com/horoscope/daily/work/{url}/{a[0][0]}/{a[0][1]}/'
            page.get(urls)
            data['工作运势'] = no_chian(BeautifulSoup(page.json['horoscope'], 'lxml').get_text())
    page.get(f'https://www.astrology.com/horoscope/daily-dating/{url}.html')
    for i in page.eles('x://*[@rel="noopener nofollow"]'):
        a = findall("""updateContent\(.*?\)""", i.html)[0]
        if get_tomorrow_date() in a:
            a = findall("""this, '(.*?)', '.*?', 'dating', '(.*?)'""", i.html)
            urls = f'https://www.astrology.com/horoscope/daily/dating/{url}/{a[0][0]}/{a[0][1]}/'
            page.get(urls)
            data['约会运势'] = no_chian(BeautifulSoup(page.json['horoscope'], 'lxml').get_text())
    return data
def nota(wedr):
    return str(wedr).replace('\xa0','')
def make_md5(s, encoding='utf-8'):
    return md5(s.encode(encoding)).hexdigest()
def no_chian(query):
    time.sleep(1)
    appid = '20240808002119047'
    appkey = '5bW5l8BO_hLMvLEq3lLn'
    salt = random.randint(32768, 65536)
    sign = make_md5(appid + query + str(salt) + appkey)
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    payload = {'appid': appid, 'q': query, 'from': 'en', 'to': 'zh', 'salt': salt, 'sign': sign}
    endpoint = 'http://api.fanyi.baidu.com'
    path = '/api/trans/vip/translate'
    url = endpoint + path
    page2.post(url, params=payload, headers=headers)
    result = page2.json
    try:
        return result['trans_result'][0]['dst']
    except:
        print(result)
    return query

def get_tomorrow_date():
    # 获取当前日期
    current_date = datetime.now()
    # 当前日期加一天
    next_day = current_date + timedelta(days=1)
    # 格式化为字符串
    return next_day.strftime('%Y-%m-%d')
def save_to_json(data):
    """将数据保存为 JSON 文件。

    参数：
    data: 要保存的数据，可以是字典或列表。
    filename: 保存的文件名，应该以 .json 结尾。
    """
    with open('data.json', 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)
if __name__ == '__main__':
    locale.setlocale(locale.LC_CTYPE, "chinese")
    dara=[]
    url=['https://www.astrology.com/horoscope/daily/aries.html', 'https://www.astrology.com/horoscope/daily/taurus.html', 'https://www.astrology.com/horoscope/daily/gemini.html', 'https://www.astrology.com/horoscope/daily/cancer.html', 'https://www.astrology.com/horoscope/daily/leo.html', 'https://www.astrology.com/horoscope/daily/virgo.html', 'https://www.astrology.com/horoscope/daily/libra.html', 'https://www.astrology.com/horoscope/daily/scorpio.html', 'https://www.astrology.com/horoscope/daily/sagittarius.html', 'https://www.astrology.com/horoscope/daily/capricorn.html', 'https://www.astrology.com/horoscope/daily/aquarius.html', 'https://www.astrology.com/horoscope/daily/pisces.html']
    page=SessionPage()
    page2 = SessionPage()
    for i in tqdm(url):
        dara.append(get(i))
    print(dara)
    save_to_json(dara)
