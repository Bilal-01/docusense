import { useRef, useState } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/Page/TextLayer.css";
import "react-pdf/dist/Page/AnnotationLayer.css";
import "./PDFViewer.css";

pdfjs.GlobalWorkerOptions.workerSrc = `https://unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

export default function PDFViewer({ file, activeChunks, onCharMapUpdate }) {
  const [numPages, setNumPages] = useState(0);
  const [scale, setScale] = useState(1.2);

  // charMap: { [pageNum]: { items: [...], pageHeight: number } }
  const charMapRef = useRef({});
  const pageRefs = useRef({});

  const onDocumentLoad = ({ numPages }) => setNumPages(numPages);

  // Extract text items with positions for a page after it renders
  const extractPageText = async (pdfPage) => {
    const viewport = pdfPage.getViewport({ scale: 1 }); // unscaled viewport
    const textContent = await pdfPage.getTextContent();

    let offset = 0;
    const items = textContent.items.map((item) => {
      const [, , , , x, y] = item.transform; // PDF coords (bottom-left origin)
      const str = item.str || "";
      const entry = {
        str,
        x,
        y,                       // PDF y from bottom of page
        width: item.width,
        height: item.height || 12,
        charStart: offset,
        charEnd: offset + str.length,
      };
      offset += str.length;
      return entry;
    });

    charMapRef.current[pdfPage.pageNumber] = {
      items,
      pageHeight: viewport.height, // unscaled page height in PDF units
    };
    onCharMapUpdate({ ...charMapRef.current });
  };

  const scrollToPage = (page) => {
    const el = pageRefs.current[page];
    if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  // Build highlight rects for all active chunks
  const highlights = {}; // { [page]: [{top, left, width, height}] }
  for (const chunk of activeChunks) {
    const page = chunk.metadata?.page_number ?? chunk.page ?? 1;
    const cs = chunk.metadata?.char_start ?? chunk.char_start ?? 0;
    const ce = chunk.metadata?.char_end ?? chunk.char_end ?? 0;
    if (ce <= cs) continue;

    const pageData = charMapRef.current[page];
    if (!pageData) continue;

    const { items, pageHeight } = pageData;
    const matched = items.filter((it) => it.charEnd > cs && it.charStart < ce);
    if (!matched.length) continue;

    if (!highlights[page]) highlights[page] = [];

    for (const rect of matched) {
      // Convert PDF bottom-up y to CSS top-down:
      // pdfTop = pageHeight - rect.y - rect.height
      // Then scale for display
      highlights[page].push({
        left:   rect.x * scale,
        top:    (pageHeight - rect.y - rect.height) * scale,
        width:  rect.width * scale,
        height: rect.height * scale,
      });
    }

    scrollToPage(page);
  }

  return (
    <div className="pdf-viewer">
      <div className="pdf-toolbar">
        <button onClick={() => setScale((s) => Math.max(0.6, s - 0.15))}>−</button>
        <span>{Math.round(scale * 100)}%</span>
        <button onClick={() => setScale((s) => Math.min(2.5, s + 0.15))}>+</button>
      </div>

      <div className="pdf-scroll">
        <Document
          file={file}
          onLoadSuccess={onDocumentLoad}
          loading={<div className="pdf-loading">Loading PDF…</div>}
          error={<div className="pdf-error">Failed to load PDF.</div>}
        >
          {Array.from({ length: numPages }, (_, i) => i + 1).map((pageNum) => (
            <div
              key={pageNum}
              className="pdf-page-wrap"
              ref={(el) => (pageRefs.current[pageNum] = el)}
            >
              <Page
                pageNumber={pageNum}
                scale={scale}
                renderAnnotationLayer={false}
                renderTextLayer={true}
                onRenderSuccess={(pdfPage) => extractPageText(pdfPage)}
              />

              {/* Highlight overlay — absolutely positioned over the canvas */}
              {(highlights[pageNum] || []).map((rect, idx) => (
                <div
                  key={idx}
                  className="pdf-highlight"
                  style={{
                    top:    rect.top,
                    left:   rect.left,
                    width:  rect.width,
                    height: rect.height,
                  }}
                />
              ))}
            </div>
          ))}
        </Document>
      </div>
    </div>
  );
}
