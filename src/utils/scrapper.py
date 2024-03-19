from requests_html import HTMLSession
from selenium import webdriver
from bs4 import BeautifulSoup
import requests as req
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.core.os_manager import ChromeType

# url = 'https://example.com'
# url = "https://fr.shopping.rakuten.com/mfp/9799998/console-sony-playstation-5-slim?pid=11945408653"
# url = "https://fr.shopping.rakuten.com/offer/buy/12389188525/boifun-babyphone-vb805.html"
# url = "https://fr.shopping.rakuten.com/offer/shop/6752062968/l-oreiller-2-en-1-qui-se-transforme-en-sac-de-couchage-medium-dimension-ouvert-137-x-50cm-gris-blanc.html?sellerLogin=Sunne"
# url = "https://fr.shopping.rakuten.com/mfp/shop/9641137/le-seigneur-des-anneaux-integrale?pid=3043714590&sellerLogin=momox&fbbaid=4499126670"
# url = "https://fr.shopping.rakuten.com/mfp/9641137/le-seigneur-des-anneaux-integrale?pid=3043714590"
# url = "https://fr.shopping.rakuten.com/offer/shop/5640308610/nendoroid-doll-harry-potter-ron-weasley-import-japonais.html?sellerLogin=GAMERZ"


# mode = 'selenium'


class RakutenScrapper():
    def __init__(self, mode='selenium'):
        self.mode = mode

    def scrap_page(self, url):
        if self.mode == 'selenium':
            driver = self.get_selenium_driver()
            driver.get(url)
            page = driver.page_source
        elif self.mode == 'HTMLSession':
            session = HTMLSession()
            response = session.get(url)
            page = response.text
        else:
            response = req.get(url)
            page = response.text

        return page

    def get_rakuten_product_infos(self, page_content):
        # your code here
        soup = BeautifulSoup(page_content, 'html.parser')
        if len(soup.select('#prdTitleHead > span.detailHeadline')) > 0:
            designation = soup.select(
                '#prdTitleHead > span.detailHeadline')[0].get_text().strip()
        else:
            designation = 'not found'
        if len(soup.select('div.fullDescription span.edito')) > 0:
            description = soup.select(
                "div.fullDescription span.edito")[0].get_text().strip()
        else:
            description = 'not found'
        if len(soup.select(".prdMainPhoto")) > 0:
            image_path = soup.select(".prdMainPhoto")[0].get('href')
        else:
            image_path = 'not found'

        if len(soup.select(".prdBreadcrumbItem")) > 0:
            true_cat = ''
            for item in soup.select(".prdBreadcrumbItem"):
                true_cat = " > ".join([true_cat, item.get_text().strip()])
        else:
            true_cat = 'not found'

        return designation, description, image_path, true_cat

    def get_selenium_driver(self):
        options = Options()
        options.add_argument("--disable-gpu")
        # options.add_argument("--headless")
        service = Service(
            ChromeDriverManager(chrome_type=ChromeType.CHROMIUM).install()
        )
        return webdriver.Chrome(
            service=service,
            options=options,
        )
