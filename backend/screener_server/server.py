from contextlib import asynccontextmanager, contextmanager, suppress
from functools import lru_cache
from threading import Lock
import logging

import requests
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from selenium.common.exceptions import WebDriverException

from credentials import email, google_api, password
from query_service import ScreenerQueryService
from schemas import RequestModel
from scraper import (
    find_company_codes,
    get_announcements,
    get_chart_data,
    get_charts,
    get_concalls,
    get_pe_charts,
    get_peers,
    get_profit_loss,
    get_quarterly_results,
    login,
    login_with_network_capture,
    run_custom_query,
)

logger = logging.getLogger(__name__)
driver_lock = Lock()
driver = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global driver
    yield
    with driver_lock:
        if driver is not None:
            with suppress(WebDriverException):
                driver.quit()
            driver = None
    if _query_service.cache_info().currsize:
        _query_service().client.close()
        _query_service.cache_clear()


app = FastAPI(title="Unified Screener API", version="1.0.0", lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str


class CompanyRequest(BaseModel):
    company_name: str
    company_id: int | None = None


@lru_cache(maxsize=1)
def _query_service():
    if not google_api:
        raise HTTPException(503, "Google API credentials are missing")
    return ScreenerQueryService(api_key=google_api)


@contextmanager
def browser():
    global driver
    if not email or not password:
        raise HTTPException(503, "Screener credentials are missing")
    with driver_lock:
        try:
            if driver is None:
                driver = login()
            yield driver
        except WebDriverException as exc:
            logger.warning("Screener browser operation failed", exc_info=exc)
            if driver is not None:
                with suppress(WebDriverException):
                    driver.quit()
                driver = None
            raise HTTPException(502, "Screener browser operation failed") from exc


def search_companies(query: str):
    try:
        response = find_company_codes(query)
        response.raise_for_status()
        companies = response.json()
        if not isinstance(companies, list):
            raise ValueError("Expected a company list")
        return companies
    except (requests.RequestException, ValueError) as exc:
        raise HTTPException(502, "Screener company search failed") from exc


def find_company(query: str):
    companies = search_companies(query)
    if not companies:
        raise HTTPException(404, "Company not found")
    return companies[0]


@app.get("/")
def read_root():
    return {
        "status": "running",
        "message": "Unified Screener API",
        "endpoints": {
            "screener": ["/login", "/search", "/charts", "/pe_charts", "/peers",
                         "/quarterly_results", "/profit_loss", "/announcements",
                         "/concalls", "/custom_query", "/chart_data"],
            "query_agent": ["/query_agent"],
        },
    }


@app.post("/login")
def login_endpoint():
    with browser():
        return {"status": "logged in"}


@app.get("/search/{query}")
def search_company(query: str):
    return search_companies(query)


@app.get("/charts/{company_url:path}")
def get_charts_endpoint(company_url: str):
    with browser() as active:
        return Response(get_charts(active, company_url), media_type="image/png")


@app.get("/pe_charts/{company_url:path}")
def get_pe_charts_endpoint(company_url: str):
    with browser() as active:
        return Response(get_pe_charts(active, company_url), media_type="image/png")


@app.get("/peers/{company_url:path}")
def get_peers_endpoint(company_url: str):
    with browser() as active:
        return {"data": get_peers(active, company_url)}


@app.get("/quarterly_results/{company_url:path}")
def get_quarterly_results_endpoint(company_url: str):
    with browser() as active:
        return {"data": get_quarterly_results(active, company_url)}


@app.get("/profit_loss/{company_url:path}")
def get_profit_loss_endpoint(company_url: str):
    with browser() as active:
        return {"data": get_profit_loss(active, company_url)}


@app.get("/announcements/{company_url:path}")
def get_announcements_endpoint(company_url: str):
    with browser() as active:
        return {"data": get_announcements(active, company_url)}


@app.get("/concalls/{company_url:path}")
def get_concalls_endpoint(company_url: str):
    with browser() as active:
        return {"links": get_concalls(active, company_url)}


@app.post("/custom_query")
def custom_query(request: QueryRequest):
    with browser() as active:
        return {"data": run_custom_query(active, request.query)}


@app.post("/accelerated_chart_data")
def accelerated_chart_data(request: CompanyRequest):
    company_id = request.company_id or find_company(request.company_name)["id"]
    url = f"https://screener.in/api/company/{company_id}/chart/"
    try:
        response = requests.get(url, params={
            "q": "Price-DMA50-DMA200-Volume", "days": 180, "consolidated": "true",
        }, timeout=(5, 15))
        response.raise_for_status()
        values = response.json()["datasets"][0]["values"]
        data = [{"label": date, "value": price} for date, price in values]
    except (requests.RequestException, IndexError, KeyError, TypeError, ValueError) as exc:
        raise HTTPException(502, "Screener chart data is unavailable") from exc

    return {
        "chartType": "line",
        "data": data,
        "xAxis": "label",
        "yAxis": ["value"],
        "title": "Past 6mo Performance",
        "description": "Price history chart of the company in last 6 months",
    }


@app.get("/chart_data/{company_url:path}")
def get_chart_data_endpoint(company_url: str):
    if not email or not password:
        raise HTTPException(503, "Screener credentials are missing")
    try:
        with driver_lock:
            active = login_with_network_capture()
            try:
                result = get_chart_data(active, company_url)
                if result:
                    return result["datasets"][0]["values"][-180:]
            finally:
                with suppress(WebDriverException):
                    active.quit()
    except (requests.RequestException, WebDriverException, IndexError, KeyError, TypeError, ValueError) as exc:
        raise HTTPException(502, "Screener chart data is unavailable") from exc
    raise HTTPException(404, "Chart data not found")


@app.post("/accelerated_concall")
def accelerated_concall(request: CompanyRequest):
    company = find_company(request.company_name)
    with browser() as active:
        links = get_concalls(active, company["url"])
    if not links:
        raise HTTPException(404, "No conference calls found")
    return links[0]


@app.post("/accelerated_peers")
def accelerated_peers(request: CompanyRequest):
    company = find_company(request.company_name)
    with browser() as active:
        return {"peers": get_peers(active, company["url"])}


@app.post("/accelerated_profit_loss")
def accelerated_profit_loss(request: CompanyRequest):
    company = find_company(request.company_name)
    with browser() as active:
        return {f"profit_loss of {company}": get_profit_loss(active, company["url"])}


@app.post("/accelerated_quarterly_results")
def accelerated_quarterly_results(request: CompanyRequest):
    company = find_company(request.company_name)
    with browser() as active:
        return {"quarterly_results": get_quarterly_results(active, company["url"])}


@app.post("/accelerated_announcements")
def accelerated_announcements(request: CompanyRequest):
    company = find_company(request.company_name)
    with browser() as active:
        return {"announcements": get_announcements(active, company["url"])}


@app.post("/query_agent")
def process_request(request: RequestModel):
    try:
        screener_query = _query_service().generate_screener_query(request.query)
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Could not generate a screener query", exc_info=exc)
        raise HTTPException(502, "Could not generate a screener query") from exc

    with browser() as active:
        output = run_custom_query(active, screener_query.screener_query)
    return {"QueryAgentResponse": output, "ScreenerQueryGenerated": screener_query}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=True)
