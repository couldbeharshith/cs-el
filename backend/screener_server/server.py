from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
import requests
import os
import time
import logging
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import from base directory
from scraper import login, login_with_network_capture, find_company_codes, get_charts, get_pe_charts, get_peers, get_quarterly_results, get_profit_loss, get_announcements, get_concalls, run_custom_query, get_chart_data
from query_service import ScreenerQueryService
from credentials import google_api
from schemas import RequestModel

app = FastAPI(title="Unified Screener API", version="1.0.0")

# Middleware for logging requests and response times
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url.path}")
    
    # Process request
    response = await call_next(request)
    
    # Calculate duration
    duration = time.time() - start_time
    
    # Log response
    logger.info(f"Response: {request.method} {request.url.path} - Status: {response.status_code} - Duration: {duration:.2f}s")
    
    return response

# Global driver instance
driver = None

# Initialize query service
query_service = ScreenerQueryService(api_key=google_api)

class QueryRequest(BaseModel):
    query: str

class CompanyRequest(BaseModel):
    company_name: str
    company_id: Optional[int] = None

def get_driver():
    global driver
    if driver is None:
        print("Driver not initialized. Logging in...")
        driver = login()
    return driver

# ============= Root =============

@app.get("/")
def read_root():
    return {
        "status": "running",
        "message": "Unified Screener API",
        "endpoints": {
            "screener": ["/login", "/search", "/charts", "/pe_charts", "/peers", 
                        "/quarterly_results", "/profit_loss", "/announcements", 
                        "/concalls", "/custom_query", "/chart_data"],
            "query_agent": ["/query_agent"]
        }
    }

# ============= Main Screener Endpoints =============

@app.post("/login")
def login_endpoint():
    logger.info("Login endpoint called")
    global driver
    if driver is None:
        driver = login()
    return {"status": "logged in"}

@app.get("/search/{query}")
def search_company(query: str):
    logger.info(f"Searching for company: {query}")
    response = find_company_codes(query)
    return response.json()

@app.get("/charts/{company_url:path}")
def get_charts_endpoint(company_url: str):
    logger.info(f"Getting charts for: {company_url}")
    driver = get_driver()
    get_charts(driver, company_url)
    if os.path.exists("chart.png"):
        return FileResponse("chart.png")
    raise HTTPException(status_code=404, detail="Chart not found")

@app.get("/pe_charts/{company_url:path}")
def get_pe_charts_endpoint(company_url: str):
    logger.info(f"Getting PE charts for: {company_url}")
    driver = get_driver()
    get_pe_charts(driver, company_url)
    if os.path.exists("chart.png"):
        return FileResponse("chart.png")
    raise HTTPException(status_code=404, detail="Chart not found")

@app.get("/peers/{company_url:path}")
def get_peers_endpoint(company_url: str):
    logger.info(f"Getting peers for: {company_url}")
    driver = get_driver()
    data = get_peers(driver, company_url)
    return {"data": data}

@app.get("/quarterly_results/{company_url:path}")
def get_quarterly_results_endpoint(company_url: str):
    logger.info(f"Getting quarterly results for: {company_url}")
    driver = get_driver()
    data = get_quarterly_results(driver, company_url)
    return {"data": data}

@app.get("/profit_loss/{company_url:path}")
def get_profit_loss_endpoint(company_url: str):
    logger.info(f"Getting profit/loss for: {company_url}")
    driver = get_driver()
    data = get_profit_loss(driver, company_url)
    return {"data": data}

@app.get("/announcements/{company_url:path}")
def get_announcements_endpoint(company_url: str):
    logger.info(f"Getting announcements for: {company_url}")
    driver = get_driver()
    data = get_announcements(driver, company_url)
    return {"data": data}

@app.get("/concalls/{company_url:path}")
def get_concalls_endpoint(company_url: str):
    logger.info(f"Getting concalls for: {company_url}")
    driver = get_driver()
    links = get_concalls(driver, company_url)
    return {"links": links}

@app.post("/custom_query")
def custom_query(request: QueryRequest):
    logger.info(f"Running custom query: {request.query[:50]}...")
    driver = get_driver()
    data = run_custom_query(driver, request.query)
    return {"data": data}

@app.post("/accelerated_chart_data")
def accelerated_chart_data(request: CompanyRequest):
    if request.company_id: 
        company_id = request.company_id
    else:
        company_id = find_company_codes(request.company_name).json()[0]["id"]
    
    url = f"https://screener.in/api/company/{company_id}/chart/?q=Price-DMA50-DMA200-Volume&days=180&consolidated=true"
    # print(url)
    res = requests.get(url).json()

    data = res["datasets"][0]["values"]
    out = {"chartType": "line",
    "data": [],
    "xAxis": "label",
    "yAxis": ["value"],
    "title": "Past 6mo Performance",
    "description": "Price history chart of the company in last 6 months"}

    for date, price in data:
        out["data"].append({"label": date, "value": price})
    
    return out

@app.get("/chart_data/{company_url:path}")
def get_chart_data_endpoint(company_url: str):
    """
    Get chart data by intercepting network requests
    Returns JSON data from the chart API
    """
    logger.info(f"Getting chart data for: {company_url}")
    start = time.time()
    
    # Use a driver with network capture enabled
    driver_with_logs = login_with_network_capture()
    login_time = time.time() - start
    logger.info(f"Login completed in {login_time:.2f}s")
    
    try:
        fetch_start = time.time()
        data = get_chart_data(driver_with_logs, company_url)
        fetch_time = time.time() - fetch_start
        logger.info(f"Data fetch completed in {fetch_time:.2f}s")
        
        if data:
            logger.info(f"Successfully retrieved chart data")
            return data["datasets"][0]["values"][-180:]
        else:
            logger.warning(f"Chart data not found for: {company_url}")
            raise HTTPException(status_code=404, detail="Chart data not found")
    finally:
        driver_with_logs.quit()
        total_time = time.time() - start
        logger.info(f"Total chart_data request time: {total_time:.2f}s")

@app.post("/accelerated_concall")
def string_call(request: CompanyRequest):
    company_name = request.company_name
    driver = get_driver()
    find_concall = find_company_codes(company_name)
    company = find_concall.json()[0]
    company_url = company['url']
    concol_link  = get_concalls(driver , company_url)
    return concol_link[0]
@app.post("/accelerated_peers")
def accelerated_peers(request: CompanyRequest):
    company_name = request.company_name
    driver = get_driver()
    find_concall = find_company_codes(company_name)
    company = find_concall.json()[0]
    company_url = company.url
    peers = get_peers(driver, company_url)
    return {"peers": peers}
@app.post("/accelerated_profit_loss")
def accelerated_profit_loss(request: CompanyRequest):
    company_name = request.company_name
    driver = get_driver()
    find_concall = find_company_codes(company_name)
    print(type(find_concall))
    print(find_concall)
    company = find_concall.json()[0]
    company_url = company['url']
    profit_loss_data = get_profit_loss(driver, company_url)
    return {f"profit_loss of {company}": profit_loss_data}
@app.post("/accelerated_quarterly_results")
def accelerated_quarterly_results(request: CompanyRequest):
    company_name = request.company_name
    driver = get_driver()
    find_concall = find_company_codes(company_name)
    company = find_concall.json()[0]
    company_url = company['url']
    quarterly_results_data = get_quarterly_results(driver, company_url)
    return {"quarterly_results": quarterly_results_data}
@app.post("/accelerated_announcements")
def accelerated_announcements(request: CompanyRequest):
    company_name = request.company_name
    driver = get_driver()
    find_concall = find_company_codes(company_name)
    company = find_concall.json()[0]
    company_url = company['url']
    announcements_data = get_announcements(driver, company_url)
    return {"announcements": announcements_data}
# ============= Query Agent Endpoint =============

@app.post("/query_agent")
def process_request(request: RequestModel):
    logger.info(f"Query agent request: {request.query[:50]}...")
    output , screener_query_generated = query_service.process_query(f'{request.query}')
    return {
        "QueryAgentResponse": output , 
        "ScreenerQueryGenerated": screener_query_generated
    }

# ============= Shutdown Handler =============

@app.on_event("shutdown")
def shutdown_event():
    global driver
    if driver:
        driver.quit()
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=True)