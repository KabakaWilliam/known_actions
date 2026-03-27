from playwright.sync_api import sync_playwright
from datetime import datetime


with sync_playwright() as p:
    browser = p.chromium.launch(
        headless=False,
        chromium_sandbox=True,
        env={},
        args=["--disable-extensions", "--disable-file-system", "--headless"],
    )
    page = browser.new_page(viewport={"width": 1280, "height": 720})
    
    # Navigate to example.com
    page.goto("https://nyt.com")
    
    # Take a screenshot and save it
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    screenshot_path = f"screenshot_{timestamp}.png"
    page.screenshot(path=screenshot_path)
    print(f"Screenshot saved to {screenshot_path}")
    
    # Close the browser
    browser.close()
