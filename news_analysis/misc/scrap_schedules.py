from bs4 import BeautifulSoup
import pandas as pd
import datetime
import requests
import os


outlets_name_id_map = {
    'London': 'p00fzl6p'
}

SCHEDULE_LINK = 'https://www.bbc.co.uk/schedules/{}/{}/{:02d}/{:02d}'
REGION_LINK = 'https://www.bbc.co.uk/schedules/p00fzl6p/2018/04/08#outlets'

def parse_schedules(schedules, date):
    rows = []
    for schedule in schedules:
        time = schedule.find_all('span', {'class': 'timezone--time'})
        #assert(len(time) == 1)
        if(len(time) > 1):
            continue
        
        time = time[0].text
        
        info = schedule.select("p[class='programme__synopsis text--subtle centi']")[0]
        info = info.find_all('span', recursive=False)
        assert(len(info) == 1)
        info = info[0].text
        
        title = schedule.find_all('span', {'class': 'programme__title delta'})
        assert(len(title) == 1)
        title = title[0].text
        
        rows.append([date, time, title, info])
    
    return rows
    
  
def scrap_schedules(start_date, end_date, timezone='London'):
    print('scraping schedule for {} timezone'.format(timezone))
    curr_date = start_date
    delta = datetime.timedelta(days=1)
    
    schedule_rows = []
    
    while curr_date <= end_date:
        print('scraping schedules for {}'.format(curr_date))
        link = SCHEDULE_LINK.format(outlets_name_id_map[timezone], curr_date.year, curr_date.month, curr_date.day)
        print(link)
        resp = requests.get(link)
        soup = BeautifulSoup(resp.text, 'html.parser')

        morning_schedules = soup.find(id='morning')
        afternoon_schedules = soup.find(id='afternoon')
        evening_schedules = soup.find(id='evening')
        late_schedules = soup.find(id='late')
        
        if morning_schedules is not None:
            morning_schedules = morning_schedules.findAll('li')
            schedule_rows += parse_schedules(morning_schedules, curr_date)
            
        if afternoon_schedules is not None:
            afternoon_schedules = afternoon_schedules.findAll('li')
            schedule_rows += parse_schedules(afternoon_schedules, curr_date)
            
        if evening_schedules is not None:
            evening_schedules = evening_schedules.findAll('li')
            schedule_rows += parse_schedules(evening_schedules, curr_date)
            
        if late_schedules is not None:
            late_schedules = late_schedules.findAll('li')
            schedule_rows += parse_schedules(late_schedules, curr_date)
        
        
        curr_date += delta
        
    schedule_df = pd.DataFrame(schedule_rows, columns=['date', 'time', 'title', 'description'])
    schedule_df.to_csv('./schedules/BBC_{}_{}_{}_show_schedules.csv'.format(timezone, start_date, end_date), index=False)
        
    
def parse_region_id_from_url(url):
    return url.split('/')[2]
    

def scrap_outlets():
    resp = requests.get(REGION_LINK)
    soup = BeautifulSoup(resp.text, 'html.parser')
    
    all_regions = soup.find('div', {'id': 'outlets'}).find_all('li')
    
    rows = [['p00fzl6p', 'London']]
    
    for region in all_regions:
        anchor = region.find_all('a')
        
        if len(anchor) == 0:
            continue
        anchor = anchor[0]
        region_id = parse_region_id_from_url(anchor['href'])
        
        rows.append([region_id, anchor.text])

    res_df = pd.DataFrame(rows, columns=['id', 'region'])
    res_df.to_csv('./regions.csv', index=False)
    

if __name__ == "__main__":
    # check if regions are scraped
    if not os.path.exists('./regions.csv'):
        scrap_outlets()
        
    regions = pd.read_csv('./regions.csv')
    
    # update region and id map
    for index, row in regions.iterrows():
        outlets_name_id_map[row['region']] = row['id']
    
    for region in outlets_name_id_map.keys():
        start_date = datetime.date(2015, 1, 1)
        end_date = datetime.date(2017, 12, 31)
        scrap_schedules(start_date, end_date, timezone=region)

