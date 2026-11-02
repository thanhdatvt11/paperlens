"""Check PDF tools without calling Ollama."""

from app.core.database import SESSION_FACTORY
from app.harness.pdf_tools import PDFTools, load_pdf_source


def main() -> None:
    """Read metadata and render one existing PDF page."""
    with SESSION_FACTORY() as db:
        source = load_pdf_source(
            db=db,
            document_id=10,
            version_id=9,
        )

    # The database session is closed before rendering.
    with PDFTools(source) as tools:
        info = tools.execute(
            "pdf_info",
            {},
        )
        print("PDF INFO:", info.data)

        result = tools.execute(
            "view_pdf_page",
            {"page": 18},
        )
        print("PAGE INFO:", result.data)
        print("IMAGE COUNT:", len(result.images))
        print("IMAGE BYTES:", len(result.images[0]))


if __name__ == "__main__":
    main()  