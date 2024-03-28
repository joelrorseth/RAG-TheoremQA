# NOTE: The rasterize_pdf function has been borrowed from the following notebook:
# https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Nougat/Inference_with_Nougat_to_read_scientific_PDFs.ipynb

import io
import fitz
from pathlib import Path
from typing import Optional, List


def get_num_pdf_pages(pdf: Path) -> int:
    return len(fitz.open(pdf))


def rasterize_pdf(
    pdf: Path,
    outpath: Optional[Path] = None,
    dpi: int = 96,
    return_pil=False,
    page_numbers=None,
) -> Optional[List[io.BytesIO]]:
    """
    Rasterize a PDF file to PNG images.

    Args:
        pdf (Path): The path to the PDF file.
        outpath (Optional[Path], optional): The output directory. If None, the PIL images will be returned instead. Defaults to None.
        dpi (int, optional): The output DPI. Defaults to 96.
        return_pil (bool, optional): Whether to return the PIL images instead of writing them to disk. Defaults to False.
        page_numbers (Optional[List[int]], optional): The pages to rasterize. If None, all pages will be rasterized. Defaults to None.

    Returns:
        Optional[List[io.BytesIO]]: The PIL images if `return_pil` is True, otherwise None.
    """

    pillow_images = []
    if outpath is None:
        return_pil = True
    try:
        if isinstance(pdf, (str, Path)):
            pdf = fitz.open(pdf)
        if page_numbers is None:
            page_numbers = range(len(pdf))
        for i in page_numbers:
            page_bytes: bytes = pdf[i-1].get_pixmap(
                dpi=dpi).pil_tobytes(format="PNG")
            if return_pil:
                pillow_images.append(io.BytesIO(page_bytes))
            else:
                with (outpath / ("%02d.png" % (i + 1))).open("wb") as f:
                    f.write(page_bytes)
    except Exception:
        pass
    if return_pil:
        return pillow_images
