import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import "./UploadZone.css";

const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

export default function UploadZone({ onSuccess, compact = false }) {
  const [status, setStatus] = useState("idle"); // idle | uploading | error
  const [progress, setProgress] = useState(0);
  const [errorMsg, setErrorMsg] = useState("");

  const onDrop = useCallback(async (accepted) => {
    const file = accepted[0];
    if (!file) return;

    setStatus("uploading");
    setProgress(0);
    setErrorMsg("");

    const formData = new FormData();
    formData.append("file", file);

    try {
      // Simulate progress while waiting (XHR gives real progress; fetch doesn't)
      const interval = setInterval(() => {
        setProgress((p) => Math.min(p + 6, 88));
      }, 300);

      const res = await fetch(`${API}/upload`, { method: "POST", body: formData });
      clearInterval(interval);

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Upload failed (${res.status})`);
      }

      const data = await res.json();
      setProgress(100);

      setTimeout(() => {
        setStatus("idle");
        setProgress(0);
        onSuccess(data.doc_id, file);
      }, 400);
    } catch (e) {
      setStatus("error");
      setErrorMsg(e.message);
    }
  }, [onSuccess]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "application/pdf": [".pdf"], "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"], "text/plain": [".txt"] },
    maxFiles: 1,
    disabled: status === "uploading",
  });

  if (compact) {
    return (
      <div {...getRootProps()} className="upload-compact">
        <input {...getInputProps()} />
        <span>↑ Replace document</span>
      </div>
    );
  }

  return (
    <div className="upload-outer">
      <div
        {...getRootProps()}
        className={`upload-zone ${isDragActive ? "drag-active" : ""} ${status === "error" ? "has-error" : ""}`}
      >
        <input {...getInputProps()} />

        {status === "uploading" ? (
          <div className="upload-progress-wrap">
            <div className="upload-progress-label">Ingesting document…</div>
            <div className="upload-progress-track">
              <div className="upload-progress-bar" style={{ width: `${progress}%` }} />
            </div>
            <div className="upload-progress-pct">{progress}%</div>
          </div>
        ) : (
          <>
            <div className="upload-icon">
              <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                <polyline points="14 2 14 8 20 8"/>
                <line x1="12" y1="18" x2="12" y2="12"/>
                <polyline points="9 15 12 12 15 15"/>
              </svg>
            </div>
            <p className="upload-title">
              {isDragActive ? "Drop to upload" : "Drop a document here"}
            </p>
            <p className="upload-sub">or click to browse — PDF, DOCX, TXT</p>
            {status === "error" && (
              <p className="upload-error">{errorMsg}</p>
            )}
          </>
        )}
      </div>
    </div>
  );
}
