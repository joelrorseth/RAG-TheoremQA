# NOTE: Most of this file is adapted from the following repository:
# https://github.com/DigitalHarborFoundation/llm-math-education

import logging
import re
import bs4
import pandas as pd
from typing import Any, Dict, List
from config import REFERENCE_HTML_TEXTBOOKS, TextbookIdentifier
from src.downloading.download_html_textbooks import load_openstax_textbook
from src.preprocessing.textbook import PreprocessedTextbook, PreprocessedTextbookSubsection
from src.utils.text import strip_excessive_whitespace


def _get_openstax_subsections(
    textbook_identifier: TextbookIdentifier, textbook_data: List[Dict[str, Any]]
) -> PreprocessedTextbook:
    textbook_data = _parse_all_openstax_sections(textbook_data)

    ds: List[PreprocessedTextbookSubsection] = []
    for chapter_section in textbook_data:
        if "subsections" not in chapter_section:
            continue
        for subsection in chapter_section["subsections"]:
            subsection = subsection.copy()
            if "subsections" in subsection and subsection["subsections"] != []:
                for subsubsection in subsection["subsections"]:
                    subsection["content"] += subsubsection["title"] + \
                        ":\n" + subsubsection["content"]
            del subsection["subsections"]
            subsection["chapter"] = chapter_section["chapter"]
            subsection["section"] = chapter_section["section"]
            ds.append(PreprocessedTextbookSubsection(**subsection))

    return PreprocessedTextbook(identifier=textbook_identifier, subsections=ds)


def _parse_all_openstax_sections(textbook_data: list[dict]) -> pd.DataFrame:
    for chapter in textbook_data:
        if chapter["section"] == 0 or type(chapter["section"]) is str:
            continue
        if "subsections" not in chapter:
            subsections, suptitle = _parse_openstax_soup(chapter["soup"])
            chapter["subsections"] = subsections
            chapter["title_text"] = suptitle
    return textbook_data


def _parse_openstax_section(section):
    header_tags = ["title", "h1", "h2", "h3", "h4", "h5"]
    content_tags = [None, "strong", "em", "b", "li", "span", "a", "sup", "u"]
    if section.has_attr("class") and "section-exercises" in section["class"]:
        return
    section_title = ""
    section_content = ""
    n_replacements = 0
    n_tables = 0
    n_images = 0
    # replace math with text representation
    # (this avoids a problem with duplicating math spans)
    for math_span in section.find_all("math"):
        math_span.replace_with(math_span.find("annotation-xml").text)
        n_replacements += 1
    # replace images
    for img in section.find_all("img"):
        # img.replace_with("Figure: " + img["alt"])
        # TODO could also include the figure number and caption without much difficulty, if desired
        img.replace_with("")
        n_replacements += 1
        n_images += 1
    subsubsections = []
    for tag in section.contents:
        text_content = None
        if tag.name == "section":
            subsubsections.append(_parse_openstax_section(tag))
            continue
        if tag.name in header_tags:
            tag_text = tag.text.strip()
            section_title += tag_text + " "
            # if tag_text.endswith("Exercises") or tag_text == "Practice Makes Perfect":
            #    break
        elif tag.name == "div":
            if tag.has_attr("data-type"):
                assert tag["data-type"] in ["note", "example", "equation"], tag
                # currently dropping this text, could save it
            elif tag.table is not None:
                n_tables += 1
                # text_content = tag.table["aria-label"]
            else:
                assert tag.figure is not None, tag
        elif type(tag) is bs4.NavigableString:
            text_content = tag.string
        else:
            assert len(tag.contents) == 1 or all(
                [child.name in content_tags for child in tag.children],
            ), f"{[child.name for child in tag.children if child.name not in content_tags]} {tag}"
            text_content = tag.text
        if text_content is not None:
            section_content += text_content + "\n"
    section_title = section_title.strip()
    section_content = re.sub(r"\n(\n)+", "\n\n", section_content.strip())
    section_content = strip_excessive_whitespace(section_content)

    return {
        "title": section_title,
        "content": section_content,
        "subsections": subsubsections,
    }


def _parse_openstax_soup(soup):
    page = soup.find(attrs={"class": "page-content"})
    suptitle = soup.find("h1").text
    # assert len(page.contents) == 1
    sections = []
    for i, section in enumerate(page.find_all("section", attrs={"data-depth": "1"})):
        parsed = _parse_openstax_section(section)
        if parsed:
            parsed["index"] = i
            sections.append(parsed)
    return sections, suptitle


def _preprocess_openstax() -> PreprocessedTextbook:
    openstax_textbook = REFERENCE_HTML_TEXTBOOKS[0]
    logging.info(f"Preprocessing HTML textbook '{openstax_textbook.name}'")
    textbook_data = load_openstax_textbook()
    return _get_openstax_subsections(openstax_textbook, textbook_data)


def preprocess_all_html_textbooks() -> List[PreprocessedTextbook]:
    return [_preprocess_openstax()]
