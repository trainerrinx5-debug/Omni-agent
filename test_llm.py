import unittest

from omni_agent.llm import OpenAICompatibleClient, _validate_base_url


class ClientConfigurationTests(unittest.TestCase):
    def test_loopback_http_and_https_are_supported(self):
        self.assertEqual(_validate_base_url("http://localhost:11434/v1"), "http://localhost:11434/v1")
        self.assertEqual(_validate_base_url("https://api.example.test/v1/"), "https://api.example.test/v1")
        self.assertTrue(OpenAICompatibleClient("https://api.example.test/v1", "model").endpoint.endswith("/v1/chat/completions"))

    def test_remote_plain_http_and_embedded_credentials_are_rejected(self):
        for url in ("http://example.test/v1", "https://user:pass@example.test/v1", "https://example.test/v1?key=x"):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    _validate_base_url(url)


if __name__ == "__main__":
    unittest.main()
