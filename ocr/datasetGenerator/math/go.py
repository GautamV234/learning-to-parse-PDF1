from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
PATH = "C:\Program Files (x86)\chromedriver.exe"
driver = webdriver.Chrome(PATH)
driver.get("https://gad.gujarat.gov.in/personnel/secretaries-government-gujarat.htm")
# print(driver.page_source)

# search = driver.find_element_by_id
# driver.quit()   

inner_container = driver.find_element_by_class_name("inner container")
print(inner_container.text)
driver.quit()
# class = inner container
