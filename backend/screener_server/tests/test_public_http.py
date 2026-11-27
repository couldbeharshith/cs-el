from unittest import TestCase
from unittest.mock import Mock, patch

import requests
from fastapi.testclient import TestClient

from server import app


class ScreenerHTTPTests(TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.company_path = "/company/EXAMPLE/consolidated/"

    def test_quarterly_results_preserve_columns_without_a_browser(self):
        company = {"id": 1, "url": self.company_path}
        page = Mock(status_code=200, is_redirect=False)
        page.text = """
            <section id="quarters">
              <h2>Quarterly Results</h2>
              <p class="sub">Consolidated Figures in Rs. Crores / <a>View Standalone</a></p>
              <table><tr><th></th><th>Jun 2026</th><th>Sep 2026</th></tr>
              <tr><td>Revenue</td><td></td><td>123</td></tr></table>
            </section>
        """
        with patch("server.search_companies", return_value=[company]), patch(
            "screener_http.requests.get", return_value=page,
        ) as get, patch("server.login") as login:
            response = self.client.post("/accelerated_quarterly_results", json={"company_name": "Example"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("Consolidated Figures in Rs. Crores", response.json()["quarterly_results"])
        self.assertIn("| Revenue |  | 123 |", response.json()["quarterly_results"])
        login.assert_not_called()
        get.assert_called_once_with(
            "https://www.screener.in/company/EXAMPLE/consolidated/", timeout=(3, 6), allow_redirects=False,
        )

    def test_reports_upstream_timeout_and_missing_table(self):
        url = "/quarterly_results" + self.company_path
        with patch("screener_http.requests.get", side_effect=requests.Timeout):
            self.assertEqual(self.client.get(url).status_code, 504)

        page = Mock(status_code=200, is_redirect=False, text="<html></html>")
        with patch("screener_http.requests.get", return_value=page):
            self.assertEqual(self.client.get(url).status_code, 404)

    def test_rejects_non_company_paths_before_fetching(self):
        with patch("screener_http.requests.get") as get:
            response = self.client.get("/quarterly_results/accounts/login")
        self.assertEqual(response.status_code, 400)
        get.assert_not_called()
