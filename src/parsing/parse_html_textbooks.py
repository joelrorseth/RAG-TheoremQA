import logging
from config import REFERENCE_HTML_TEXTBOOKS


def parse_openstax():
    # Note: Openstax preprocessing uses raw html data, this is a no-op
    openstax_textbook = REFERENCE_HTML_TEXTBOOKS[0]
    logging.info(f"Parsing HTML textbook '{openstax_textbook.name}'")


def parse_all_html_textbooks():
    parse_openstax()
