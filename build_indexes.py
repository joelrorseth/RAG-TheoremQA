import logging
from src.downloading.download_html_textbooks import download_all_html_textbooks
from src.downloading.download_pdf_textbooks import download_all_pdf_textbooks
from src.indexing.vector import build_all_indexes_for_all_strategies
from src.parsing.parse_html_textbooks import parse_all_html_textbooks
from src.parsing.parse_pdf_textbooks import parse_all_pdf_textbooks
# from src.preprocessing.preprocess_html_textbooks import preprocess_all_html_textbooks
from src.preprocessing.preprocess_pdf_textbooks import preprocess_all_pdf_textbooks


def build_all_indexes():
    logging.info("[DOWNLOAD] Downloading all textbooks")
    # download_all_html_textbooks()
    download_all_pdf_textbooks()
    logging.info("[DOWNLOAD] Successfully downloaded all textbooks")

    logging.info("[PARSE] Parsing all textbooks")
    # parse_all_html_textbooks()
    parse_all_pdf_textbooks()
    logging.info("[PARSE] Successfully parsed all textbooks")

    logging.info("[PREPROCESS] Preprocessing all textbooks")
    # TODO: Fix HTML preprocessing file, which is currently broken
    # html_textbooks = preprocess_all_html_textbooks()
    all_textbooks = preprocess_all_pdf_textbooks()
    logging.info("[PREPROCESS] Successfully preprocessed all textbooks")

    logging.info("[INDEXING] Building all indexes")
    build_all_indexes_for_all_strategies(all_textbooks)
    logging.info("[INDEXING] Successfully built all indexes")


build_all_indexes()
