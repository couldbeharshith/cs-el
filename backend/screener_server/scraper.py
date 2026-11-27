from contextlib import suppress
import os
from urllib.parse import urlencode

from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from credentials import email, password
from screener_http import SCREENER_ORIGIN, company_page_url


PAGE_TIMEOUT = 12
ELEMENT_TIMEOUT = 6
LOGIN_URL = f"{SCREENER_ORIGIN}/login/"
CHART = (By.CSS_SELECTOR, "#canvas-chart-holder canvas")
QUERY_TABLE = (By.CSS_SELECTOR, "main table")


def login():
    options = webdriver.ChromeOptions()
    options.page_load_strategy = "eager"
    options.add_argument("--headless=new")
    options.add_experimental_option("prefs", {"profile.managed_default_content_settings.images": 2})
    if chrome_major := os.getenv("SCREENER_CHROME_MAJOR"):
        options.browser_version = chrome_major
    return sign_in(webdriver.Chrome(options=options))


def sign_in(driver):
    try:
        driver.set_page_load_timeout(PAGE_TIMEOUT)
        driver.get(LOGIN_URL)
        driver.find_element(By.ID, "id_username").send_keys(email)
        password_field = driver.find_element(By.ID, "id_password")
        password_field.send_keys(password)
        previous = driver.current_url
        password_field.send_keys(Keys.RETURN)
        WebDriverWait(driver, ELEMENT_TIMEOUT).until(EC.url_changes(previous))
        return driver
    except Exception:
        with suppress(WebDriverException):
            driver.quit()
        raise


def get_charts(driver, company_url):
    driver.get(company_page_url(company_url) + "#chart")
    canvas = WebDriverWait(driver, ELEMENT_TIMEOUT).until(EC.visibility_of_element_located(CHART))
    return canvas.screenshot_as_png


def get_pe_charts(driver, company_url):
    driver.get(company_page_url(company_url) + "#chart")
    wait = WebDriverWait(driver, ELEMENT_TIMEOUT)
    button = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="company-chart-metrics"]//button[normalize-space()="PE Ratio"]')))
    button.click()
    canvas = wait.until(EC.visibility_of_element_located(CHART))
    return canvas.screenshot_as_png


def run_custom_query(driver, query):
    params = urlencode({"query": query, "limit": 100})
    driver.get(f"{SCREENER_ORIGIN}/screen/raw/?{params}")
    if "/login/" in driver.current_url or "/register/" in driver.current_url:
        raise WebDriverException("Screener session expired")

    try:
        return WebDriverWait(driver, ELEMENT_TIMEOUT).until(EC.visibility_of_element_located(QUERY_TABLE)).text
    except TimeoutException:
        return "No results found. The query might be too restrictive or the website is temporarily unavailable."
