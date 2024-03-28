# NOTE: _download_or_load_openstax_textbook is adapted from the following repository:
# https://github.com/DigitalHarborFoundation/llm-math-education

import logging
import time
from typing import Any, Dict, List
import bs4
import requests
from config import (
    REFERENCE_HTML_DATA_PATH,
    REFERENCE_HTML_TEXTBOOKS,
    TextbookIdentifier
)

RETRIEVAL_DELAY_S = 0.25


def _download_or_load_openstax_textbook(
    textbook: TextbookIdentifier, overwrite: bool = False
) -> List[Dict[str, Any]]:
    download_dir_path = REFERENCE_HTML_DATA_PATH / textbook.name
    download_dir_path.mkdir(exist_ok=True)

    intro_filepath = download_dir_path / "intro.html"
    if intro_filepath.exists() and not overwrite:
        with open(intro_filepath) as infile:
            html_doc = infile.read()
        soup = bs4.BeautifulSoup(html_doc, "html.parser")
    else:
        data = requests.get(textbook.source_url)
        html_doc = data.content.decode()
        soup = bs4.BeautifulSoup(html_doc, "html.parser")
        with open(intro_filepath, "w") as outfile:
            outfile.write(soup.prettify())
    toc = soup.find_all(attrs={"class": "os-text"})
    assert len(toc) > 0, "Unexpected table-of-contents structure."

    # parse URLs from the table of contents
    textbook_data = []
    toc_links = toc[3].parent.parent.parent.parent.parent.parent.find_all("a")
    for link in toc_links:
        href = link["href"]
        url_tokens = href.split("-")
        try:
            chapter = int(url_tokens[0])
        except ValueError:
            continue
        try:
            section = int(url_tokens[1])
        except ValueError:
            section = 0
        if section == 0:
            for expected_section_name in ["key-terms", "key-concepts", "review-exercises", "practice-test"]:
                if href.endswith(expected_section_name):
                    section = expected_section_name.replace("-", "_")
        if section == 0:
            continue
        textbook_data.append(
            {
                "chapter": chapter,
                "section": section,
                "href": href,
            },
        )
    # retrieve textbook data page by page
    root_url = textbook.source_url.split("pages")[0] + "pages/"
    for i, chapter_data in enumerate(textbook_data):
        part_filepath = download_dir_path / f"part{i}.html"
        if part_filepath.exists() and not overwrite:
            with open(part_filepath) as infile:
                html_doc = infile.read()
            soup = bs4.BeautifulSoup(html_doc, "html.parser")
        else:
            data = requests.get(root_url + chapter_data["href"])
            html_doc = data.content.decode()
            soup = bs4.BeautifulSoup(html_doc, "html.parser")
            with open(part_filepath, "w") as outfile:
                outfile.write(soup.prettify())
            time.sleep(RETRIEVAL_DELAY_S)
        chapter_data["soup"] = soup
    return textbook_data


def download_openstax_textbook():
    openstax_textbook = REFERENCE_HTML_TEXTBOOKS[0]
    logging.info(f"Downloading HTML textbook '{openstax_textbook.name}'")
    _download_or_load_openstax_textbook(openstax_textbook)


def load_openstax_textbook() -> List[Dict[str, Any]]:
    openstax_textbook = REFERENCE_HTML_TEXTBOOKS[0]
    return _download_or_load_openstax_textbook(openstax_textbook)


def download_all_html_textbooks():
    download_openstax_textbook()
