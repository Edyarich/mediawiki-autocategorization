#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Собираем датасет с 0x1.tv
"""
import pickle

import mwclient


class Dataset40x1tv:
    """
    Собираем датасет с сайта.
    """

    def __init__(self):
        self.dataset = {}
        self.dataset = {
            "articles": [],
            "categories": [],
        }

        self.site = mwclient.Site("0x1.tv", scheme="https", path="/")
        self.site.login("bot", "discopal")

    def process(self, save_path: str = "0x1tv-dataset.pickle"):
        def get_article_for_page(page):
            article = {"title": page.page_title, "text": page.text(), "categories": []}
            for cat in page.categories():
                article["categories"].append(cat.name)
            return article

        dsc = self.dataset["categories"]
        dsa = self.dataset["articles"]

        for page in self.site.allpages(namespace=0):
            if not page.redirect:
                print("Processing", page.page_title)
                dsa.append(get_article_for_page(page))

        for page in self.site.allpages(namespace=14):
            if not page.redirect:
                print("Processing", page.page_title)
                dsc.append(get_article_for_page(page))

        pickle.Pickler(open(save_path, "wb")).dump(self.dataset)
