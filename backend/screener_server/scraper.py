import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
import time
import requests
import urllib.parse
import json
from credentials import email, password

url = "https://www.screener.in/login/"

def login():
    # Specify version_main to match your Chrome browser version (143)
    driver = uc.Chrome(headless=True, use_subprocess=False, version_main=143)
    driver.get(url)
    time.sleep(3)
    driver.find_element(By.XPATH, '//*[@id="id_username"]').send_keys(email)
    driver.find_element(By.XPATH, '//*[@id="id_password"]').send_keys(password)
    driver.find_element(By.XPATH, '//*[@id="id_password"]').send_keys(Keys.RETURN)
    time.sleep(5)
    return driver

def login_with_network_capture():
    """Login with network request capture enabled - optimized for speed"""
    options = uc.ChromeOptions()
    
    # Enable performance logging
    options.set_capability('goog:loggingPrefs', {'performance': 'ALL'})
    
    # Performance optimizations
    prefs = {
        'profile.managed_default_content_settings.images': 2,  # Disable images
        'profile.managed_default_content_settings.stylesheets': 2,  # Disable CSS
        'profile.managed_default_content_settings.cookies': 1,
        'profile.managed_default_content_settings.javascript': 1,
        'profile.managed_default_content_settings.plugins': 2,
        'profile.managed_default_content_settings.popups': 2,
        'profile.managed_default_content_settings.geolocation': 2,
        'profile.managed_default_content_settings.notifications': 2,
        'profile.managed_default_content_settings.media_stream': 2,
    }
    options.add_experimental_option('prefs', prefs)
    
    # Additional speed optimizations
    options.add_argument('--disable-gpu')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-extensions')
    options.add_argument('--disable-blink-features=AutomationControlled')
    
    # Specify version_main to match your Chrome browser version (143)
    driver = uc.Chrome(options=options, use_subprocess=False, headless=True, version_main=143)
    driver.get(url)
    print("URL Fetched")
    time.sleep(3) 
    driver.find_element(By.XPATH, '//*[@id="id_username"]').send_keys(email)
    driver.find_element(By.XPATH, '//*[@id="id_password"]').send_keys(password)
    driver.find_element(By.XPATH, '//*[@id="id_password"]').send_keys(Keys.RETURN)
    time.sleep(4)
    return driver

def find_company_codes(query):
    url = f"https://www.screener.in/api/company/search/?q={query}"
    response = requests.get(url)
    print(response.json())
    return response

def get_charts(driver, company_url):
    url = f"https://www.screener.in/{company_url}#chart"
    driver.get(url)
    time.sleep(3)
    canvas = driver.find_element(By.XPATH, '//*[@id="canvas-chart-holder"]/canvas')
    canvas.screenshot("chart.png")

def get_pe_charts(driver, company_url):
    url = f"https://www.screener.in/{company_url}#chart"
    driver.get(url)
    time.sleep(3)
    driver.find_element(By.XPATH, '//*[@id="company-chart-metrics"]/button[2]').click()
    canvas = driver.find_element(By.XPATH, '//*[@id="canvas-chart-holder"]/canvas')
    canvas.screenshot("chart.png")

def get_peers(driver, company_url):
    url = f"https://www.screener.in/{company_url}"
    driver.get(url)
    time.sleep(3)
    peers_table = driver.find_element(By.XPATH, '//*[@id="peers-table-placeholder"]')
    return peers_table.text

def get_quarterly_results(driver, company_url):
    url = f"https://www.screener.in/{company_url}/#quarters"
    driver.get(url)
    time.sleep(5)
    quarterly_results = driver.find_element(By.XPATH, '//*[@id="quarters"]')
    print(quarterly_results.text)
    return quarterly_results.text

def get_profit_loss(driver, company_url):
    url = f"https://www.screener.in/{company_url}"
    driver.get(url)
    time.sleep(3)
    profit_loss = driver.find_element(By.XPATH, '//*[@id="profit-loss"]')
    print(profit_loss.text)
    return profit_loss.text

def get_announcements(driver, company_url):
    url = f"https://www.screener.in/{company_url}"
    driver.get(url)
    time.sleep(3)
    driver.find_element(By.XPATH, '//*[@id="documents"]/div[2]/div[4]/div[2]/button/i').click()
    time.sleep(2)
    announcements = driver.find_element(By.XPATH, '//*[@id="documents"]/div[2]/div[1]/div')
    print(announcements.text)
    return announcements.text

def get_concalls(driver, company_url):
    url = f"https://www.screener.in/{company_url}"
    driver.get(url)
    time.sleep(3)
    concalls = driver.find_element(By.XPATH, '//*[@id="documents"]/div[2]/div[4]')
    links = concalls.find_elements(By.TAG_NAME, 'a')
    download_links = []
    for link in links:
        href = link.get_attribute('href')
        if href:
            download_links.append(href)
    return download_links

def run_custom_query(driver, query):
    encoded_query = urllib.parse.quote(query)
    url = f"https://www.screener.in/screen/raw/?query={encoded_query}&limit=100"
    driver.get(url)
    time.sleep(8)  # Increased wait time
    
    try:
        # Try specific XPath first
        table = driver.find_element(By.XPATH, '/html/body/main/div[2]/div[4]/table')
    except Exception:
        print(f"Specific XPath failed. Current URL: {driver.current_url}")
        try:
            # Try finding any table
            table = driver.find_element(By.TAG_NAME, 'table')
        except Exception as e:
            # If still fails, return a friendly error message
            print(f"Could not find table. Page might be loading or query returned no results.")
            return "No results found. The query might be too restrictive or the website is temporarily unavailable."
    
    print(table.text)
    return table.text

def get_chart_data(driver, company_url):
    """
    Intercept network requests to get chart API data
    Returns the JSON data from the chart API endpoint
    """
    # Navigate to the company page
    full_url = f"https://www.screener.in/{company_url}"
    driver.get(full_url)
    
    # Wait for API calls with shorter timeout
    time.sleep(2)  # Reduced from 5
    
    # Get performance logs to capture network requests
    logs = driver.get_log('performance')
    
    chart_api_url = None
    
    # Parse logs to find chart API requests (optimized - break early)
    for log in logs:
        try:
            log_data = json.loads(log['message'])
            message = log_data.get('message', {})
            method = message.get('method', '')
            
            # Look for network response received events
            if method == 'Network.responseReceived':
                params = message.get('params', {})
                response = params.get('response', {})
                request_url = response.get('url', '')
                
                # Check if this is a chart API request
                if '/api/company/' in request_url and '/chart/' in request_url:
                    chart_api_url = request_url
                    print(f"Found chart API URL: {chart_api_url}")
                    break
        except:
            continue
    
    if chart_api_url:
        # Get cookies from the driver to make authenticated request
        cookies = driver.get_cookies()
        session = requests.Session()
        for cookie in cookies:
            session.cookies.set(cookie['name'], cookie['value'])
        
        # Make request to the chart API
        response = session.get(chart_api_url)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to fetch chart data. Status code: {response.status_code}")
            return None
    else:
        print("Chart API URL not found in network logs")
        return None
