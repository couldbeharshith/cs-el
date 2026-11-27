from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import Mock, patch

from selenium.common.exceptions import TimeoutException, WebDriverException

from scraper import sign_in
from server import app, lifespan


class ScreenerLifecycleTests(IsolatedAsyncioTestCase):
    async def test_shutdown_closes_resources_when_lifespan_raises(self):
        browser = Mock()
        browser.quit.side_effect = WebDriverException("browser already closed")
        service = Mock()
        service.cache_info.return_value.currsize = 1
        with patch("server.driver", browser), patch("server._query_service", service):
            with self.assertRaisesRegex(RuntimeError, "lifespan failure"):
                async with lifespan(app):
                    raise RuntimeError("lifespan failure")
        browser.quit.assert_called_once_with()
        service.return_value.client.close.assert_called_once_with()
        service.cache_clear.assert_called_once_with()


class SignInTests(TestCase):
    def test_failed_cleanup_does_not_hide_login_error(self):
        browser = Mock()
        browser.get.side_effect = TimeoutException("login timed out")
        browser.quit.side_effect = WebDriverException("browser already closed")
        with self.assertRaisesRegex(TimeoutException, "login timed out"):
            sign_in(browser)
        browser.quit.assert_called_once_with()
