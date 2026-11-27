from contextlib import asynccontextmanager, contextmanager, suppress
from functools import lru_cache
from threading import Lock
import logging

import requests
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from selenium.common.exceptions import TimeoutException, WebDriverException

from credentials import email, google_api, password
from query_service import ScreenerQueryService
from schemas import RequestModel
from scraper import get_charts, get_pe_charts, login, run_custom_query
from screener_http import (
    InvalidCompanyPath,
    MissingCompanyData,
    fetch_announcements,
    fetch_chart_values,
    fetch_company_chart,
    fetch_concalls,
    fetch_financial_table,
    fetch_peers,
    find_company_codes,
)

logger = logging.getLogger(__name__)
driver_lock = Lock()
driver = None


def close_browser():
    global driver
    active, driver = driver, None
    if active is not None:
        with suppress(WebDriverException):
            active.quit()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        yield
    finally:
        try:
            with driver_lock:
                close_browser()
        finally:
            if _query_service.cache_info().currsize:
                try:
                    _query_service().client.close()
                finally:
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
    if not driver_lock.acquire(timeout=3):
        raise HTTPException(503, "Screener browser is busy")
    try:
        if driver is None:
            driver = login()
        yield driver
    except TimeoutException as exc:
        close_browser()
        raise HTTPException(504, "Screener browser operation timed out") from exc
    except WebDriverException as exc:
        logger.warning("Screener browser operation failed", exc_info=exc)
        close_browser()
        raise HTTPException(502, "Screener browser operation failed") from exc
    finally:
        driver_lock.release()


def search_companies(query: str):
    try:
        response = find_company_codes(query)
        response.raise_for_status()
        companies = response.json()
        if not isinstance(companies, list):
            raise ValueError("Expected a company list")
        return companies
    except requests.Timeout as exc:
        raise HTTPException(504, "Screener company search timed out") from exc
    except (requests.RequestException, ValueError) as exc:
        raise HTTPException(502, "Screener company search failed") from exc


def find_company(query: str):
    companies = search_companies(query)
    if not companies:
        raise HTTPException(404, "Company not found")
    return companies[0]


def public_data(fetch, *args):
    try:
        return fetch(*args)
    except InvalidCompanyPath as exc:
        raise HTTPException(400, "Expected a Screener company path") from exc
    except MissingCompanyData as exc:
        raise HTTPException(404, str(exc)) from exc
    except requests.Timeout as exc:
        raise HTTPException(504, "Screener request timed out") from exc
    except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as exc:
        raise HTTPException(502, "Screener company data is unavailable") from exc


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
        return Response(public_data(get_charts, active, company_url), media_type="image/png")


@app.get("/pe_charts/{company_url:path}")
def get_pe_charts_endpoint(company_url: str):
    with browser() as active:
        return Response(public_data(get_pe_charts, active, company_url), media_type="image/png")


@app.get("/peers/{company_url:path}")
def get_peers_endpoint(company_url: str):
    return {"data": public_data(fetch_peers, company_url)}


@app.get("/quarterly_results/{company_url:path}")
def get_quarterly_results_endpoint(company_url: str):
    return {"data": public_data(fetch_financial_table, company_url, "quarters")}


@app.get("/profit_loss/{company_url:path}")
def get_profit_loss_endpoint(company_url: str):
    return {"data": public_data(fetch_financial_table, company_url, "profit-loss")}


@app.get("/announcements/{company_url:path}")
def get_announcements_endpoint(company_url: str):
    return {"data": public_data(fetch_announcements, company_url)}


@app.get("/concalls/{company_url:path}")
def get_concalls_endpoint(company_url: str):
    return {"links": public_data(fetch_concalls, company_url)}


@app.post("/custom_query")
def custom_query(request: QueryRequest):
    with browser() as active:
        return {"data": run_custom_query(active, request.query)}


@app.post("/accelerated_chart_data")
def accelerated_chart_data(request: CompanyRequest):
    company_id = request.company_id or find_company(request.company_name)["id"]
    values = public_data(fetch_chart_values, company_id, True, 180)
    data = [{"label": date, "value": price} for date, price in values]
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
    return public_data(fetch_company_chart, company_url)[-180:]


@app.post("/accelerated_concall")
def accelerated_concall(request: CompanyRequest):
    company = find_company(request.company_name)
    links = public_data(fetch_concalls, company["url"])
    if not links:
        raise HTTPException(404, "No conference calls found")
    return links[0]


@app.post("/accelerated_peers")
def accelerated_peers(request: CompanyRequest):
    company = find_company(request.company_name)
    return {"peers": public_data(fetch_peers, company["url"])}


@app.post("/accelerated_profit_loss")
def accelerated_profit_loss(request: CompanyRequest):
    company = find_company(request.company_name)
    return {f"profit_loss of {company}": public_data(fetch_financial_table, company["url"], "profit-loss")}


@app.post("/accelerated_quarterly_results")
def accelerated_quarterly_results(request: CompanyRequest):
    company = find_company(request.company_name)
    return {"quarterly_results": public_data(fetch_financial_table, company["url"], "quarters")}


@app.post("/accelerated_announcements")
def accelerated_announcements(request: CompanyRequest):
    company = find_company(request.company_name)
    return {"announcements": public_data(fetch_announcements, company["url"])}


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
    uvicorn.run(app, host="127.0.0.1", port=8080)
