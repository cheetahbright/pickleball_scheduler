# streamlit_gui_autotest.py
# Selenium script to automatically click all buttons in a Streamlit app and check for errors in the GUI.
# Usage: pip install selenium webdriver-manager
# Then: python streamlit_gui_autotest.py

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.service import Service as FirefoxService
from webdriver_manager.firefox import GeckoDriverManager
import time

# CONFIG
STREAMLIT_URL = "http://localhost:8501"  # Change if your Streamlit app runs elsewhere
WAIT_TIME = 1.5  # seconds to wait after each click

# Start browser
options = webdriver.FirefoxOptions()
# options.add_argument('--headless')  # Remove this line if you want to see the browser

browser = webdriver.Firefox(service=FirefoxService(GeckoDriverManager().install()), options=options)
browser.get(STREAMLIT_URL)
time.sleep(3)  # Wait for app to load

# Click all visible buttons and check for errors
buttons = browser.find_elements(By.TAG_NAME, "button")
clicked = set()
for i, btn in enumerate(buttons):
    try:
        label = btn.text.strip()
        if not label or btn in clicked:
            continue
        # Skip if not displayed or not enabled
        if not btn.is_displayed() or not btn.is_enabled():
            continue
        print(f"Clicking button: {label}")
        # Scroll into view
        browser.execute_script("arguments[0].scrollIntoView({block: 'center'});", btn)
        time.sleep(0.3)
        try:
            btn.click()
        except Exception as e:
            print(f"Standard click failed for '{label}', trying JS click. Reason: {e}")
            try:
                browser.execute_script("arguments[0].click();", btn)
            except Exception as e2:
                print(f"JS click also failed for '{label}': {e2}")
                continue
        clicked.add(btn)
        time.sleep(WAIT_TIME)
        # Attempt to close overlays if present (tutorial, modals, etc.)
        try:
            close_btns = browser.find_elements(By.XPATH, "//button[contains(., 'Exit Tutorial') or contains(., 'Close') or contains(., 'Cancel')]")
            for cbtn in close_btns:
                if cbtn.is_displayed() and cbtn.is_enabled():
                    print("Closing overlay/modal...")
                    browser.execute_script("arguments[0].scrollIntoView({block: 'center'});", cbtn)
                    time.sleep(0.2)
                    try:
                        cbtn.click()
                    except Exception:
                        browser.execute_script("arguments[0].click();", cbtn)
                    time.sleep(0.5)
        except Exception as e:
            print(f"Error trying to close overlays: {e}")
        # Check for Streamlit error messages
        error_divs = browser.find_elements(By.CSS_SELECTOR, '[data-testid="stExceptionDetails"]')
        if error_divs:
            print(f"ERROR after clicking '{label}': {error_divs[0].text}")
        else:
            print(f"No error after clicking '{label}'")
    except Exception as e:
        print(f"Exception clicking button '{getattr(btn, 'text', '?')}': {e}")

print("\nDone. Closing browser.")
browser.quit()
