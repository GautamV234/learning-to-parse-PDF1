from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import pandas as pd
import numpy as np
import cv2
import pyautogui

driver = webdriver.Chrome()
driver.get("https://atomurl.net/math/")



data = pd.read_csv('data.csv')
print(data.head)
formulae = data['latex']
names = data['image']
inputElement = driver.find_element_by_class_name("textareaedit")
print(len(formulae))
for name,formula in zip(names,formulae):
    formula = str(formula)
    formula = formula.strip()
    print(f"formula is {formula}")
    inputElement.send_keys(formula)
    inputElement.send_keys(Keys.ENTER)
    image = pyautogui.screenshot()
    print("Screenshot taken")
    image = cv2.cvtColor(np.array(image),
                     cv2.COLOR_RGB2BGR)
    name = name.split(".")[0]
    final_name = name+".png"
    cv2.imwrite(final_name, image)
    # cv2.imshow('image',image)
    inputElement.clear()
    
