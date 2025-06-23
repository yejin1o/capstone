#Import Selenium webdriver library
from selenium import webdriver
#Set virtual browser to chrome
driver = webdriver.Chrome('/opt/google/chrome/chromedriver')
#Run query to select Keyword text
sql = "select keyword from instagram"
#Exploring Instagram pages using search keyword
line = "https://www.instagram.com/explore/tags/"+ txt2
driver.get(line)
#Page scrollbar down to full
driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
#Crawling the user id of postings that contain search terms
userid = driver.find_element_by_css_selector("user_id").text
#Save user id
sql = "insert into instagram set userid= %s"
#Run query to select saved user id
sql = "select no, userid from useridBOX where state = %s order by no desc"
#Exploring instagram pages using user id
line = "https://www.instagram.com/"+ userid
driver.get(line)
#Page scrollbar down to full
driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
#Crawls all links to user-generated postings.
allList = find_element_by_tag_name('a').get_attribute("href")
#Use the link to navigate to the page and crawl the Instagram data.
for aurl in allList:
    driver.get(aurl)
    #Instagram image
    image = driver.find_element_by_css_selector("img").get_attribute("src")
    #Instagram date
    date = driver.find_element_by_css_selector("date").get_attribute("datetime")
    #Instagram contents
    content = driver.find_element_by_class_name("content").text
    #Save data
    sql = "insert into instagram set imgae = %s, date = %s, content = %s"