import logging
from math import ceil
from pathlib import Path
from typing import Generator, List, Set
from transformers import NougatProcessor, VisionEncoderDecoderModel
from PIL import Image
from config import (
    HF_OCR_MODEL_DEVICE,
    HF_OCR_MODEL_NAME,
    MAX_OCR_PARALLEL_PAGES,
    MAX_OCR_TOKENS_PER_PAGE,
    REFERENCE_PDF_PAGE_FILENAME_PREFIX,
    REFERENCE_PDF_PAGE_FILENAME_SUFFIX,
    REFERENCE_PDF_PAGES_SUBDIR_NAME,
    REFERENCE_PDF_TEXTBOOK_FILENAME,
    REFERENCE_PDF_TEXTBOOKS,
    REFERENCE_PDFS_PATH,
    TextbookIdentifier
)
from src.utils.rasterize import get_num_pdf_pages, rasterize_pdf


def get_saved_page_numbers(directory_path: Path) -> Set[int]:
    """
    Scans the given directory for files matching the page file pattern, and
    returns a list of integers i found in the file names.
    """
    page_numbers = set()

    for file in directory_path.iterdir():
        # Check if the file matches the page file pattern
        if (
            file.name.startswith(REFERENCE_PDF_PAGE_FILENAME_PREFIX)
            and file.name.endswith(REFERENCE_PDF_PAGE_FILENAME_SUFFIX)
        ):
            try:
                # Extract the number part of the file name and convert to int
                page_number = int(file.name[len(
                    REFERENCE_PDF_PAGE_FILENAME_PREFIX):
                    -len(REFERENCE_PDF_PAGE_FILENAME_SUFFIX)
                ])
                page_numbers.add(page_number)
            except ValueError:
                continue

    return page_numbers


def generate_textbook_page_texts(
    textbook: TextbookIdentifier, pdf_path: Path, pages_dir_path: Path,
    overwrite: bool = False
) -> Generator[str, None, None]:
    """
    Generates text for each page of a specified PDF, and saves each page as a text file.
    Unless `overwrite == True`, pages that have already been saved will not be re-read.
    """
    # Figure out which pages need parsing
    num_pdf_pages = get_num_pdf_pages(pdf_path)
    saved_page_numbers = get_saved_page_numbers(pages_dir_path)
    all_page_numbers = set([i+1 for i in range(num_pdf_pages)])
    page_numbers_to_parse = list(
        (all_page_numbers - saved_page_numbers)
        if overwrite == False
        else all_page_numbers
    )
    num_pages_to_parse = len(page_numbers_to_parse)
    num_page_batches = ceil(num_pages_to_parse / MAX_OCR_PARALLEL_PAGES)

    if num_pages_to_parse == 0:
        logging.info(f"Already parsed PDF textbook '{textbook.name}', skipping")
        return

    logging.info("Loading Nougat OCR model")
    processor = NougatProcessor.from_pretrained(HF_OCR_MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(HF_OCR_MODEL_NAME)
    model.to(HF_OCR_MODEL_DEVICE)

    batch_counter = 0
    for batch_start_idx in range(0, num_pages_to_parse, MAX_OCR_PARALLEL_PAGES):
        batch_counter += 1
        logging.info(
            f"Parsing PDF pages batch {batch_counter}/{num_page_batches}"
        )

        # Get the images for this batch
        page_numbers_batch = page_numbers_to_parse[
            batch_start_idx:batch_start_idx + MAX_OCR_PARALLEL_PAGES
        ]
        images = rasterize_pdf(
            pdf_path, page_numbers=page_numbers_batch, return_pil=True
        )
        selected_images = [Image.open(img) for img in images]
        pixel_values = processor(
            images=selected_images, return_tensors="pt"
        ).pixel_values

        # Batch inference on selected pages
        outputs = model.generate(
            pixel_values.to(HF_OCR_MODEL_DEVICE),
            min_length=1,
            max_new_tokens=MAX_OCR_TOKENS_PER_PAGE,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
        )

        texts = processor.batch_decode(outputs, skip_special_tokens=True)
        texts = processor.post_process_generation(texts, fix_markdown=False)
        for text_idx, text in enumerate(texts):
            page_num = page_numbers_batch[text_idx]
            yield page_num, text


def parse_pdf_textbook(textbook: TextbookIdentifier):
    """
    Parse a single reference PDF textbook.
    """
    logging.info(f"Parsing PDF textbook '{textbook.name}'")
    textbook_dir_path = REFERENCE_PDFS_PATH / textbook.name
    textbook_pdf_path = textbook_dir_path / REFERENCE_PDF_TEXTBOOK_FILENAME

    if not textbook_dir_path.is_dir() or not textbook_pdf_path.exists():
        logging.error(
            f"Could not find textbook to parse at {textbook_dir_path}")
        return

    # Create subdirectory for page text files in the textbook directory
    pages_dir_path = textbook_dir_path / REFERENCE_PDF_PAGES_SUBDIR_NAME
    pages_dir_path.mkdir(exist_ok=True)
    logging.info(f"Parsing pages in {textbook_pdf_path}")

    for page_num, page_text in generate_textbook_page_texts(
        textbook, textbook_pdf_path, pages_dir_path
    ):
        # Write each page text as a separate text file
        page_file_path = pages_dir_path / \
            f"{REFERENCE_PDF_PAGE_FILENAME_PREFIX}{page_num}{REFERENCE_PDF_PAGE_FILENAME_SUFFIX}"
        with open(page_file_path, "w") as file:
            file.write(page_text)


def parse_pdf_textbooks(textbooks: List[TextbookIdentifier]):
    """
    Parse the given reference PDF textbooks.
    """
    for textbook in textbooks:
        parse_pdf_textbook(textbook)


def parse_all_pdf_textbooks():
    parse_pdf_textbooks(REFERENCE_PDF_TEXTBOOKS)
