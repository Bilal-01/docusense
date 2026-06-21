import { useState } from "react";
import PDFViewer from "./components/PDFViewer";
import ChatPanel from "./components/ChatPanel";
import UploadZone from "./components/UploadZone";
import "./App.css";

export default function App() {
  const [docId, setDocId] = useState(null);
  const [pdfFile, setPdfFile] = useState(null);
  const [activeChunks, setActiveChunks] = useState([]);
  const [charMap, setCharMap] = useState({});

  const handleUploadSuccess = (id, file) => {
    setDocId(id);
    setPdfFile(file);
    setActiveChunks([]);
  };

  return (
    <div className="app">
      <header className="app-header">
        <div className="logo">
          <span className="logo-mark">D</span>
          <span className="logo-text">DocuSense</span>
        </div>
        <span className="logo-tagline">Source-attributed document intelligence</span>
      </header>

      <main className="app-body">
        <section className="panel panel-left">
          {pdfFile ? (
            <PDFViewer
              file={pdfFile}
              activeChunks={activeChunks}
              onCharMapUpdate={setCharMap}
            />
          ) : (
            <UploadZone onSuccess={handleUploadSuccess} />
          )}
          {pdfFile && (
            <div className="replace-upload">
              <UploadZone onSuccess={handleUploadSuccess} compact />
            </div>
          )}
        </section>

        <section className="panel panel-right">
          <ChatPanel
            docId={docId}
            charMap={charMap}
            onHighlight={setActiveChunks}
          />
        </section>
      </main>
    </div>
  );
}
