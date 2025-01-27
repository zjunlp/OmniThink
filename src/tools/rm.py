import logging
import os
from typing import Callable, Union, List
import dspy
import requests
import re
import uuid
import json
from src.utils.WebPageHelper import WebPageHelper


def clean_text(res):
    pattern = r'\[.*?\]\(.*?\)'
    result = re.sub(pattern, '', res)
    url_pattern = pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    result = re.sub(url_pattern, '', result)
    result = re.sub(r"\n\n+", "\n", result)
    return result

class GoogleSearchAli(dspy.Retrieve):
    def __init__(self, bing_search_api_key=None, k=3, is_valid_source: Callable = None,
                 min_char_count: int = 150, snippet_chunk_size: int = 1000, webpage_helper_max_threads=10,
                 mkt='en-US', language='en-US', **kwargs):

        super().__init__(k=k)
        key = os.environ.get('SEARCHKEY', 'default_value')
        self.header = {
            "Content-Type": "application/json",
            "Accept-Encoding": "utf-8",
            "Authorization": f"Bearer lm-/{key}== ",
        }

        self.template = {
            "rid": str(uuid.uuid4()),
            "scene": "dolphin_search_bing_nlp",
            "uq": "",
            "debug": True,
            "fields": [],
            "page": 1,
            "rows": 10,
            "customConfigInfo": {
                "multiSearch": False,
                "qpMultiQuery": False,
                "qpMultiQueryHistory": [],
                "qpSpellcheck": False,
                "qpEmbedding": False,
                "knnWithScript": False,
                "qpTermsWeight": False,
                "pluginServiceConfig": {"qp": "mvp_search_qp_qwen"},  # v3 rewrite
            },
            "headers": {"__d_head_qto": 5000},
        }
        
        self.webpage_helper = WebPageHelper(
            min_char_count=min_char_count,
            snippet_chunk_size=snippet_chunk_size,
            max_thread_num=webpage_helper_max_threads
        )
        self.usage = 0

        # If not None, is_valid_source shall be a function that takes a URL and returns a boolean.
        if is_valid_source:
            self.is_valid_source = is_valid_source
        else:
            self.is_valid_source = lambda x: True

    def get_usage_and_reset(self):
        usage = self.usage
        self.usage = 0

        return {'BingSearch': usage}

    def forward(self, query_or_queries: Union[str, List[str]], exclude_urls: List[str] = []):

        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        self.usage += len(queries)

        url_to_results = {}

        for query in queries:
            try:
                self.template["uq"] = query

                response = requests.post(
                    "https://nlp-cn-beijing.aliyuncs.com/gw/v1/api/msearch-sp/qwen-search",
                    data=json.dumps(self.template),
                    headers=self.header,
                )              
                response = json.loads(response.text)
                search_results = response['data']['docs']
                for result in search_results:
                    url_to_results[result['url']] = {
                        'url': result['url'],
                        'title': result['title'],
                        'description': result.get('snippet', '')
                    }
            except Exception as e:
                logging.error(f'Error occurs when searching query {query}: {e}')

        valid_url_to_snippets = self.webpage_helper.urls_to_snippets(list(url_to_results.keys()))
        collected_results = []
        for url in valid_url_to_snippets:
            r = url_to_results[url]
            r['snippets'] = valid_url_to_snippets[url]['snippets']
            collected_results.append(r)

        print(f'lengt of collected_results :{len(collected_results)}')
        return collected_results
    

class BingSearchAli(dspy.Retrieve):
    def __init__(self, bing_search_api_key=None, k=3, is_valid_source: Callable = None,
                 min_char_count: int = 150, snippet_chunk_size: int = 1000, webpage_helper_max_threads=10,
                 mkt='en-US', language='en-US', **kwargs):
        """
        Params:
            min_char_count: Minimum character count for the article to be considered valid.
            snippet_chunk_size: Maximum character count for each snippet.
            webpage_helper_max_threads: Maximum number of threads to use for webpage helper.
            mkt, language, **kwargs: Bing search API parameters.
            - Reference: https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/reference/query-parameters
        """
        super().__init__(k=k)
        if not bing_search_api_key and not os.environ.get("SEARCH_ALI_API_KEY"):
            raise RuntimeError(
                "You must supply bing_search_api_key or set environment variable SEARCH_ALI_API_KEY")
        elif bing_search_api_key:
            self.bing_api_key = bing_search_api_key
        else:
            self.bing_api_key = os.environ["SEARCH_ALI_API_KEY"]
        self.endpoint = "https://idealab.alibaba-inc.com/api/v1/search/search"
        self.count = k
        self.params = {
            'mkt': mkt,
            "setLang": language,
            "count": k,
            **kwargs
        }
        self.webpage_helper = WebPageHelper(
            min_char_count=min_char_count,
            snippet_chunk_size=snippet_chunk_size,
            max_thread_num=webpage_helper_max_threads
        )
        self.usage = 0

        # If not None, is_valid_source shall be a function that takes a URL and returns a boolean.
        if is_valid_source:
            self.is_valid_source = is_valid_source
        else:
            self.is_valid_source = lambda x: True

    def get_usage_and_reset(self):
        usage = self.usage
        self.usage = 0

        return {'BingSearch': usage}

    def forward(self, query_or_queries: Union[str, List[str]], exclude_urls: List[str] = []):
        """Search with Bing for self.k top passages for query or queries

        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.
            exclude_urls (List[str]): A list of urls to exclude from the search results.

        Returns:
            a list of Dicts, each dict has keys of 'description', 'snippets' (list of strings), 'title', 'url'
        """
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        self.usage += len(queries)

        url_to_results = {}

        payload_template = {
            "query": "pleaceholder",
            "num": self.count,
            "extendParams": {
                "country": "US",
                "locale": "en-US",
                "location": "United States",
                "page": 2
            },
            "platformInput": {
                "model": "google-search",
                "instanceVersion": "S1"
            }
        }
        header = {"X-AK": self.bing_api_key, "Content-Type": "application/json"}

        for query in queries:
            try:
                payload_template["query"] = query
                response = requests.post(
                    self.endpoint,
                    headers=header,
                    json=payload_template,
                ).json()
                search_results = response['data']['originalOutput']['webPages']['value']

                for result in search_results:
                    url_to_results[result['url']] = {
                        'url': result['url'],
                        'title': result['name'],
                        'description': result.get('snippet', '')
                    }
            except Exception as e:
                logging.error(f'Error occurs when searching query {query}: {e}')

        valid_url_to_snippets = self.webpage_helper.urls_to_snippets(list(url_to_results.keys()))
        collected_results = []
        for url in valid_url_to_snippets:
            r = url_to_results[url]
            r['snippets'] = valid_url_to_snippets[url]['snippets']
            collected_results.append(r)
        return collected_results


class BingSearch(dspy.Retrieve):
    def __init__(self, bing_search_api_key=None, k=3, is_valid_source: Callable = None,
                 min_char_count: int = 150, snippet_chunk_size: int = 1000, webpage_helper_max_threads=10,
                 mkt='en-US', language='en', **kwargs):
        """
        Params:
            min_char_count: Minimum character count for the article to be considered valid.
            snippet_chunk_size: Maximum character count for each snippet.
            webpage_helper_max_threads: Maximum number of threads to use for webpage helper.
            mkt, language, **kwargs: Bing search API parameters.
            - Reference: https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/reference/query-parameters
        """
        super().__init__(k=k)
        if not bing_search_api_key and not os.environ.get("BING_SEARCH_API_KEY"):
            raise RuntimeError(
                "You must supply bing_search_subscription_key or set environment variable BING_SEARCH_API_KEY")
        elif bing_search_api_key:
            self.bing_api_key = bing_search_api_key
        else:
            self.bing_api_key = os.environ["BING_SEARCH_API_KEY"]
        self.endpoint = "https://api.bing.microsoft.com/v7.0/search"
        self.params = {
            'mkt': mkt,
            "setLang": language,
            "count": k,
            **kwargs
        }
        self.webpage_helper = WebPageHelper(
            min_char_count=min_char_count,
            snippet_chunk_size=snippet_chunk_size,
            max_thread_num=webpage_helper_max_threads
        )
        self.usage = 0

        # If not None, is_valid_source shall be a function that takes a URL and returns a boolean.
        if is_valid_source:
            self.is_valid_source = is_valid_source
        else:
            self.is_valid_source = lambda x: True

    def get_usage_and_reset(self):
        usage = self.usage
        self.usage = 0

        return {'BingSearch': usage}

    def forward(self, query_or_queries: Union[str, List[str]], exclude_urls: List[str] = []):
        """Search with Bing for self.k top passages for query or queries

        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.
            exclude_urls (List[str]): A list of urls to exclude from the search results.

        Returns:
            a list of Dicts, each dict has keys of 'description', 'snippets' (list of strings), 'title', 'url'
        """
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        self.usage += len(queries)

        url_to_results = {}

        headers = {"Ocp-Apim-Subscription-Key": self.bing_api_key , "Content-Type": "application/json" }

        for query in queries:
            try:
                results = requests.get(
                    self.endpoint,
                    headers=headers,
                    params={**self.params, 'q': query}
                ).json()

                for d in results['webPages']['value']:
                    if self.is_valid_source(d['url']) and d['url'] not in exclude_urls:
                        url_to_results[d['url']] = {'url': d['url'], 'title': d['name'], 'description': d['snippet']}
            except Exception as e:
                logging.error(f'Error occurs when searching query {query}: {e}')

        valid_url_to_snippets = self.webpage_helper.urls_to_snippets(list(url_to_results.keys()))
        collected_results = []
        for url in valid_url_to_snippets:
            r = url_to_results[url]
            r['snippets'] = valid_url_to_snippets[url]['snippets']
            collected_results.append(r)
        return collected_results


class SearXNG(dspy.Retrieve):
    def __init__(
        self,
        searxng_api_url,
        searxng_api_key=None,
        k=3,
        is_valid_source: Callable = None,
    ):
        """Initialize the SearXNG search retriever.
        Please set up SearXNG according to https://docs.searxng.org/index.html.

        Args:
            searxng_api_url (str): The URL of the SearXNG API. Consult SearXNG documentation for details.
            searxng_api_key (str, optional): The API key for the SearXNG API. Defaults to None. Consult SearXNG documentation for details.
            k (int, optional): The number of top passages to retrieve. Defaults to 3.
            is_valid_source (Callable, optional): A function that takes a URL and returns a boolean indicating if the
            source is valid. Defaults to None.
        """
        super().__init__(k=k)
        if not searxng_api_url:
            raise RuntimeError("You must supply searxng_api_url")
        self.searxng_api_url = searxng_api_url
        self.searxng_api_key = searxng_api_key
        self.usage = 0

        if is_valid_source:
            self.is_valid_source = is_valid_source
        else:
            self.is_valid_source = lambda x: True

    def get_usage_and_reset(self):
        usage = self.usage
        self.usage = 0
        return {"SearXNG": usage}

    def forward(
        self, query_or_queries: Union[str, List[str]], exclude_urls: List[str] = []
    ):
        """Search with SearxNG for self.k top passages for query or queries

        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.
            exclude_urls (List[str]): A list of urls to exclude from the search results.

        Returns:
            a list of Dicts, each dict has keys of 'description', 'snippets' (list of strings), 'title', 'url'
        """
        
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        self.usage += len(queries)
        collected_results = []
        headers = (
            {"Authorization": f"Bearer {self.searxng_api_key}"}
            if self.searxng_api_key
            else {}
        )

        for query in queries:
            try:
                params = {"q": query, "format": "json"}
                response = requests.get(
                    self.searxng_api_url, headers=headers, params=params, #categories="science"
                )
                results = response.json()

                for r in results["results"]:
                    if self.is_valid_source(r["url"]) and r["url"] not in exclude_urls:
                        collected_results.append(
                            {
                                "description": r.get("content", ""),
                                "snippets": [r.get("content", "")],
                                "title": r.get("title", ""),
                                "url": r["url"],
                            }
                        )
            except Exception as e:
                logging.error(f"Error occurs when searching query {query}: {e}")

        return collected_results