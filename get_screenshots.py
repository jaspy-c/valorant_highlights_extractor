import os
import argparse
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Scrape a VLR.gg match page and take screenshots.")
    parser.add_argument("url", type=str, help="The URL of the VLR.gg match page")
    args = parser.parse_args()
    
    # Define the directory where screenshots will be saved
    screenshots_dir = "screenshots"
    os.makedirs(screenshots_dir, exist_ok=True)
    
    # Configure Chrome options (using headless mode here, remove if you want to see the browser)
    chrome_options = webdriver.ChromeOptions()
    # chrome_options.add_argument("--headless")
    
    # Set up the WebDriver (update 'chromedriver.exe' with the actual path if needed)
    service = Service('chromedriver.exe')
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    try:
        # Maximize the browser window to fullscreen
        driver.maximize_window()
        
        # Navigate to the provided URL
        driver.get(args.url)
        
        # Locate the dark mode toggle button
        dark_mode_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CLASS_NAME, "header-switch.js-dark-switch"))
        )
        dark_mode_button.click()
        time.sleep(2)
        
        # Wait for a stable element after toggling dark mode
        element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "vm-stats-container"))
        )
        driver.execute_script("arguments[0].scrollIntoView(true);", element)
        time.sleep(1)
        
        # Get all map elements that are NOT disabled (data-disabled="0")
        map_items = WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CLASS_NAME, "vm-stats-gamesnav-item.js-map-switch"))
        )

        # Filter out only the ones where data-disabled="0"
        map_items = [item for item in map_items if item.get_attribute("data-disabled") == "0"]

        num_maps = len(map_items)
        
        # Loop through each map and take a screenshot
        for i in range(num_maps):
            map_item = map_items[i]
            map_href = map_item.get_attribute("data-href")
            print(f"Clicking map {i} (href: {map_href})")
            map_item.click()
            
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "vm-stats-container"))
            )
            
            screenshot_path = os.path.join(screenshots_dir, f"map_{i if i != 0 else 'all'}_screenshot.png")
            element = driver.find_element(By.CLASS_NAME, "vm-stats-container")
            element.screenshot(screenshot_path)
            print(f"Screenshot for map {i} saved to {screenshot_path}")
    
    finally:
        driver.quit()
        
if __name__ == "__main__":
    main()
