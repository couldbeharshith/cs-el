import base64
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()


@dataclass
class QueryResult:
    success: bool
    fields: List[str] | None = None
    values: List[list] | None = None
    error: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None


class AuraQueryAPI:
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        # uri examples: neo4j+s://bd614210.databases.neo4j.io
        host = uri.replace("neo4j+s://", "").replace("neo4j://", "").replace("bolt://", "")
        self.url = f"https://{host}/db/{database}/query/v2"
        token = base64.b64encode(f"{username}:{password}".encode()).decode()
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Basic {token}"
        }

    def execute(self, statement: str, parameters: Optional[Dict[str, Any]] = None, timeout: int = 30) -> QueryResult:
        payload = {"statement": statement, "parameters": parameters or {}}
        try:
            resp = requests.post(self.url, headers=self.headers, json=payload, timeout=timeout)
            if resp.status_code in (200, 202):
                data = resp.json()
                d = data.get("data", {})
                return QueryResult(True, d.get("fields"), d.get("values"), None, data)
            return QueryResult(False, None, None, f"HTTP {resp.status_code}: {resp.text}")
        except Exception as e:
            return QueryResult(False, None, None, str(e))

    def ping(self) -> bool:
        r = self.execute("RETURN 1 as ok")
        return r.success and r.fields == ["ok"] and r.values and r.values[0] == [1]


def _demo():
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER", "neo4j")
    pwd = os.getenv("NEO4J_PASSWORD")
    client = AuraQueryAPI(uri, user, pwd)
    print(f"Endpoint: {client.url}")
    test = client.ping()
    print("Ping:", test)

if __name__ == "__main__":
    _demo()
