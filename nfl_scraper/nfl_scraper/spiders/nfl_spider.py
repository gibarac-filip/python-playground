import scrapy
import csv

class NflItem(scrapy.Item):
    title = scrapy.Field()
    content = scrapy.Field()

class NflSpider(scrapy.Spider):
    name = 'nfl_spider'
    start_urls = ['https://www.nfl.com/games/lions-at-chiefs-2023-reg-1']  # Start URL(s) for crawling

    '''
    def parse(self, response):
        # Extract data here and populate NflItem instances
        item = NflItem()
        item['title'] = response.css('h1::text').get()
        item['content'] = response.css('p::text').get()
        yield item

        # Follow links to other pages if needed
        for next_page in response.css('a::attr(href)'):
            yield response.follow(next_page, self.parse)
    '''
    def parse(self, response):
        # Extract data here and check for the word "punt"
        text_elements = response.css('p::text').getall()

        # Filter and save elements containing "punt"
        punt_elements = [element.strip() for element in text_elements if "punt" in element.lower()]

        if punt_elements:
            # Save to CSV
            with open('punt_data.csv', 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['Text'])
                for element in punt_elements:
                    csv_writer.writerow([element])

        # Follow links to other pages if needed
        for next_page in response.css('a::attr(href)'):
            yield response.follow(next_page, self.parse)

