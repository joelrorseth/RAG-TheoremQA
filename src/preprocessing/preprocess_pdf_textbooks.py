import json
import logging
from difflib import get_close_matches
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Generator, List
from config import (
    INDEX_CHUNK_SIZE,
    REFERENCE_PDF_CONTENTS_FILENAME,
    REFERENCE_PDF_OUTLINE_FILENAME,
    REFERENCE_PDF_PAGE_FILENAME_PREFIX,
    REFERENCE_PDF_PAGE_FILENAME_SUFFIX,
    REFERENCE_PDF_PAGES_SUBDIR_NAME,
    REFERENCE_PDF_TEXTBOOKS,
    REFERENCE_PDFS_PATH
)
from src.preprocessing.textbook import PreprocessedTextbook, PreprocessedTextbookChunk


CHAPTER_HEADER_DEPTH = 2
SECTION_HEADER_DEPTH = 3


def _get_outline_dict(textbook_name: str) -> Dict[str, Any]:
    outline_json_path = REFERENCE_PDFS_PATH / \
        textbook_name / REFERENCE_PDF_OUTLINE_FILENAME

    with open(outline_json_path, 'r') as file:
        outline_dict = json.load(file)

    return outline_dict


def _generate_page_paths(
    pages_directory_path: Path, start_page: int, end_page: int
) -> Generator[Path, None, None]:
    """
    Generate Path objects for each persisted page file in a given directory.
    """
    page_num = start_page
    while page_num <= end_page:
        page_filename = f"{REFERENCE_PDF_PAGE_FILENAME_PREFIX}{page_num}{REFERENCE_PDF_PAGE_FILENAME_SUFFIX}"
        page_path = pages_directory_path / page_filename
        if page_path.is_file():
            yield page_path
            page_num += 1
        else:
            break


def _read_textbook_page_lines(textbook_page_path: Path) -> List[str]:
    """
    Read the lines of a textbook page from file into a list of strings.
    """
    try:
        with open(textbook_page_path, 'r', encoding='utf-8') as file:
            file_content = file.read()
            return file_content.split("\n")

    except ValueError as e:
        logging.error(e)
        return ""


def read_textbook_lines(
    pages_directory_path: Path, contents_dict: Dict[str, Any]
) -> List[str]:
    """
    Read the lines of a textbook from file into a list of strings.
    """
    start_page = contents_dict["startpage"]
    end_page = contents_dict["endpage"]
    return [
        page_line
        for page_file_path in _generate_page_paths(pages_directory_path, start_page, end_page)
        for page_line in _read_textbook_page_lines(page_file_path)
    ]


def _parse_and_write_pdf_textbook_contents(
    textbook_name: str, textbook_content_file_path: Path
):
    """
    Parse a PDF textbook from page files, saving the result into contents.jsonl
    """
    def alpha_only(mystr: str) -> str:
        return ''.join(
            char
            for char in mystr
            if char.isalpha() and char.lower() in "abcdefghijklmnopqrstuvwxyz"
        )

    def is_header(depth: int, target_header_title: str, target_header_number: str, line: str) -> bool:
        if not line.startswith(f"{'#' * depth} "):
            return False
        alpha_only_target = target_header_title.lower().strip()
        alpha_only_cur_line = alpha_only(line.lower().strip())
        alpha_only_cur_line = alpha_only_cur_line.replace("chapter", "")
        is_close_alpha_match = len(get_close_matches(
            alpha_only_target,
            [alpha_only_cur_line],
            cutoff=0.9
        )) > 0
        is_section_num_match = line.startswith(
            f"{'#' * depth} {target_header_number} ")
        is_new_section = is_section_num_match or is_close_alpha_match
        return is_new_section

    # Load all lines of the textbook into a list
    logging.info(f"Parsing and writing PDF textbook '{textbook_name}'")
    textbook_pages_directory_path = REFERENCE_PDFS_PATH / \
        textbook_name / REFERENCE_PDF_PAGES_SUBDIR_NAME
    textbook_outline_dict = _get_outline_dict(textbook_name)
    textbook_lines = read_textbook_lines(
        textbook_pages_directory_path, textbook_outline_dict
    )

    # Create list of chapter and section checkpoints in expected order
    chapter_section_targets = []
    for chapter_idx, chapter in enumerate(textbook_outline_dict["chapters"]):
        chapter_section_targets.append((chapter_idx, chapter, None, None))
        for section_idx, section in enumerate(chapter["sections"]):
            chapter_section_targets.append(
                (chapter_idx, chapter, section_idx, section))

    line_idx = 0
    cur_section_lines: List[str] = []
    chapter_content = []
    contents = []
    target_idx = 0

    # Quick stats
    # words_per_line = [len(ll.split()) for ll in textbook_lines]
    # print(f"AVG: {mean(words_per_line)}")
    # print(f"MED: {median(words_per_line)}")
    # print(f"MIN: {min(words_per_line)}")
    # print(f"MAX: {max(words_per_line)}")

    while line_idx < len(textbook_lines):
        # Unpack the next expected (aka "target") chapter / section
        (chapter_idx, chapter, section_idx, section) = (
            chapter_section_targets[target_idx]
            if target_idx < len(chapter_section_targets)
            else (None, None, None, None)
        )

        cur_line: str = textbook_lines[line_idx].strip()
        line_idx += 1

        # Skip empty lines
        if len(cur_line) == 0:
            continue

        # Check if this line starts a new chapter
        if ((chapter is not None and section is None) and is_header(
            CHAPTER_HEADER_DEPTH, chapter["title"], f"Chapter {chapter_idx+1}", cur_line
        )):
            logging.info(f"Found Chapter {chapter_idx+1}: {cur_line}")
            target_idx += 1
            if chapter_idx == 0:
                cur_section_lines = [cur_line]
            else:
                chapter_content.append(cur_section_lines)
                contents.append(chapter_content)
                chapter_content = []
                cur_section_lines = [cur_line]

        # Check if this line starts a new section
        elif (section is not None and is_header(
            SECTION_HEADER_DEPTH, section["title"], f"{chapter_idx+1}.{section_idx+1}", cur_line
        )):
            logging.info(
                f"Found Section {chapter_idx+1}.{section_idx+1}: {cur_line}")
            target_idx += 1
            chapter_content.append(cur_section_lines)
            cur_section_lines = [cur_line]

        else:
            cur_section_lines.append(cur_line)

    # Flush remaining content
    if len(cur_section_lines) > 0:
        chapter_content.append(cur_section_lines)
        cur_section_lines = []

    if len(chapter_content) > 0:
        contents.append(chapter_content)
        chapter_content = []

    # Verify that the expected number of chapters were read
    # TODO: Also check that last chapter has expected number of sections
    num_chapters = len(textbook_outline_dict["chapters"])
    if len(contents) < num_chapters:
        raise ValueError(
            f"Only parsed {len(contents)}/{num_chapters} chapters"
        )

    # Write the content to a contents file
    textbook_content_file_path = REFERENCE_PDFS_PATH / \
        textbook_name / REFERENCE_PDF_CONTENTS_FILENAME
    with open(textbook_content_file_path, 'w') as file:
        json.dump(contents, file)


def _get_textbook_line_chunks(
    textbook_contents: List[List[List[str]]]
) -> List[PreprocessedTextbookChunk]:
    chunks = []
    for chapter_idx, chapter in enumerate(textbook_contents):
        for section_idx, section in enumerate(chapter):
            for section_line_idx, section_line in enumerate(section):
                words = section_line.split()
                for i in range(0, len(words), INDEX_CHUNK_SIZE):
                    line_chunk = " ".join(words[i:i+INDEX_CHUNK_SIZE])
                    chunks.append(PreprocessedTextbookChunk(
                        chapter_idx=chapter_idx,
                        section_idx=section_idx,
                        subsection_idx=0,
                        line_idx=section_line_idx,
                        content=line_chunk
                    ))

    return chunks


def preprocess_all_pdf_textbooks() -> List[PreprocessedTextbook]:
    preprocessed_textbooks = []

    for textbook in REFERENCE_PDF_TEXTBOOKS:
        logging.info(f"Preprocessing PDF textbook '{textbook.name}'")
        textbook_content_file_path = REFERENCE_PDFS_PATH / \
            textbook.name / REFERENCE_PDF_CONTENTS_FILENAME

        if not textbook_content_file_path.exists():
            _parse_and_write_pdf_textbook_contents(
                textbook.name, textbook_content_file_path
            )

        with open(textbook_content_file_path, 'r') as file:
            textbook_contents = json.load(file)
            preprocessed_textbooks.append(
                PreprocessedTextbook(
                    identifier=textbook,
                    chunks=_get_textbook_line_chunks(textbook_contents)
                )
            )

    return preprocessed_textbooks
