# -*- coding: utf-8 -*-
import scrapy


class ImagebotSpider(scrapy.Spider):
    name = 'imagebot'
    allowed_domains = ['www.allelectronics.com/category/140/capacitors/1.html']
    start_urls = ['http://www.allelectronics.com/category/140/capacitors/1.html']

    def parse(self, response):
        #Extract product information
        images = response.css("img.thumb_image::attr(data-src)").extract()




        for item in images:
            scraped_info = {

                'image_urls' : ["https://www.allelectronics.com" + x for x in item], #Set's the url for scrapy to download images
            }

        yield scraped_info
