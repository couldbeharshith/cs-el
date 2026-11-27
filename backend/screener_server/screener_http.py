import re
from urllib.parse import unquote, urljoin, urlsplit

import requests
from bs4 import BeautifulSoup


class InvalidCompanyPath(ValueError):
    pass


SCREENER_ORIGIN = "https://www.screener.in"


class MissingCompanyData(ValueError):
    pass


def company_page_url(company_url: str) -> str:
    try:
        parsed = urlsplit(company_url)
        path = unquote(parsed.path).strip("/")
    except ValueError as exc:
        raise InvalidCompanyPath(company_url) from exc

    if (
        (parsed.netloc and parsed.netloc.lower() not in {"www.screener.in", "screener.in"})
        or (parsed.scheme and parsed.scheme not in {"http", "https"})
        or not re.fullmatch(r"company/[A-Za-z0-9_&.-]+(?:/consolidated)?", path)
        or path.split("/")[1] in {".", ".."}
    ):
        raise InvalidCompanyPath(company_url)

    return f"{SCREENER_ORIGIN}/{path}/"


def company_page(company_url: str) -> BeautifulSoup:
    response = requests.get(company_page_url(company_url), timeout=(3, 6), allow_redirects=False)
    if response.status_code == 404:
        raise MissingCompanyData("Company not found")
    response.raise_for_status()
    if response.is_redirect:
        raise requests.HTTPError("Unexpected redirect from Screener")

    return BeautifulSoup(response.text, "html.parser")


def fetch_financial_table(company_url: str, section_id: str) -> str:
    document = company_page(company_url)
    section = document.find(id=section_id)
    table = section.find("table") if section else None
    if table is None:
        raise MissingCompanyData("Screener financial table not found")

    rows = [
        [cell.get_text(" ", strip=True).replace("|", r"\|") for cell in row.find_all(["th", "td"], recursive=False)]
        for row in table.find_all("tr")
    ]
    rows = [row for row in rows if row]
    if len(rows) < 2 or len(rows[0]) < 2 or any(len(row) != len(rows[0]) for row in rows):
        raise MissingCompanyData("Screener financial table not found")

    heading = section.find("h2")
    units = section.find("p", class_="sub")
    intro = [
        node.get_text(" ", strip=True).split(" / ", 1)[0]
        for node in (heading, units)
        if node
    ]
    rows[0][0] = rows[0][0] or "Metric"
    rows.insert(1, ["---"] * len(rows[0]))
    lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n\n".join([*intro, "\n".join(lines)])



def fetch_peers(company_url: str) -> str:
    container = company_page(company_url).find(id="peers-table-placeholder")
    if container is None or container.find("table") is None:
        raise MissingCompanyData("Screener peer table not found")
    return container.find("table").get_text("\n", strip=True)


def fetch_announcements(company_url: str) -> str:
    container = company_page(company_url).find(id="company-announcements-tab")
    if container is None:
        raise MissingCompanyData("Screener announcements not found")
    return container.get_text("\n", strip=True)


def fetch_concalls(company_url: str) -> list[str]:
    document = company_page(company_url)
    for heading in document.select("#documents .documents h3"):
        if heading.get_text(strip=True).casefold() == "concalls":
            links = [
                urljoin(SCREENER_ORIGIN, anchor["href"])
                for anchor in heading.find_parent("div", class_="documents").select("a[href]")
                if anchor["href"].strip() and not anchor["href"].startswith("#")
            ]
            return list(dict.fromkeys(link for link in links if urlsplit(link).scheme in {"https", "http"}))
    return []


def fetch_chart_values(company_id: int, consolidated: bool, days: int = 365) -> list:
    response = requests.get(
        f"{SCREENER_ORIGIN}/api/company/{company_id}/chart/",
        params={"q": "Price-DMA50-DMA200-Volume", "days": days, "consolidated": str(consolidated).lower()},
        timeout=(3, 6),
        allow_redirects=False,
    )
    if response.status_code == 404:
        raise MissingCompanyData("Chart data not found")
    response.raise_for_status()
    datasets = response.json()["datasets"]
    values = datasets[0]["values"] if datasets else []
    if not values:
        raise MissingCompanyData("Chart data not found")
    if not isinstance(values, list) or any(not isinstance(point, list) or len(point) != 2 for point in values):
        raise ValueError("Invalid Screener chart response")
    return values


def fetch_company_chart(company_url: str) -> list:
    info = company_page(company_url).find(id="company-info")
    if info is None or not (info.get("data-company-id") or "").isdigit():
        raise MissingCompanyData("Chart data not found")
    return fetch_chart_values(int(info["data-company-id"]), info.get("data-consolidated") == "true")


def find_company_codes(query: str):
    return requests.get(
        f"{SCREENER_ORIGIN}/api/company/search/",
        params={"q": query},
        timeout=(3, 6),
    )
