import { useState } from "react";
import "./MessageBubble.css";

export default function MessageBubble({ data, onHighlight }) {
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const [activeIds, setActiveIds] = useState([]);

  const { answer = [], raw_answer, faithfulness_score, answer_relevancy, source_chunks = [] } = data;

  // Build a map: chunk_id → source_chunk for quick lookup
  const chunkById = {};
  for (const c of source_chunks) chunkById[c.chunk_id] = c;

  const handleSentenceEnter = (chunkIds) => {
    setActiveIds(chunkIds);
    const chunks = chunkIds.map((id) => chunkById[id]).filter(Boolean);
    onHighlight(chunks);
  };

  const handleSentenceLeave = () => {
    setActiveIds([]);
    onHighlight([]);
  };

  const handleSourceClick = (chunk) => {
    onHighlight([chunk]);
  };

  const faithColor =
    faithfulness_score == null ? null
    : faithfulness_score >= 0.85 ? "faith-green"
    : faithfulness_score >= 0.7  ? "faith-yellow"
    : "faith-red";

  const faithTooltip =
    faithfulness_score != null && faithfulness_score < 0.7
      ? "This answer may contain claims not fully supported by the source document."
      : null;

  // Collect unique footnote indices for chunk IDs
  const footnotesMap = {}; // chunk_id → number
  let fnCounter = 1;
  for (const item of answer) {
    for (const id of item.chunk_ids || []) {
      if (!(id in footnotesMap)) footnotesMap[id] = fnCounter++;
    }
  }

  return (
    <div className="message-bubble">
      {/* Faithfulness badge */}
      {faithfulness_score != null && (
        <div className={`faith-badge ${faithColor}`} title={faithTooltip || ""}>
          {faithTooltip && <span className="faith-warn">⚠ </span>}
          Faithfulness {(faithfulness_score * 100).toFixed(0)}%
          {answer_relevancy != null && (
            <span className="faith-rel"> · Relevancy {(answer_relevancy * 100).toFixed(0)}%</span>
          )}
        </div>
      )}

      {/* Answer sentences with footnote superscripts */}
      <div className="bubble-answer">
        {answer.map((item, i) => {
          const ids = item.chunk_ids || [];
          const isActive = ids.some((id) => activeIds.includes(id));
          const hasSource = ids.length > 0 && ids[0] !== "None";

          return (
            <span
              key={i}
              className={`sentence ${hasSource ? "has-source" : ""} ${isActive ? "sentence-active" : ""}`}
              onMouseEnter={hasSource ? () => handleSentenceEnter(ids) : undefined}
              onMouseLeave={hasSource ? handleSentenceLeave : undefined}
            >
              {item.sentence}{" "}
              {hasSource && ids.map((id) =>
                footnotesMap[id] ? (
                  <sup key={id} className="footnote-badge">{footnotesMap[id]}</sup>
                ) : null
              )}
            </span>
          );
        })}
      </div>

      {/* Sources collapsible */}
      {source_chunks.length > 0 && (
        <div className="sources-section">
          <button
            className="sources-toggle"
            onClick={() => setSourcesOpen((o) => !o)}
          >
            <span>{sourcesOpen ? "▾" : "▸"} Sources</span>
            <span className="sources-count">{source_chunks.length}</span>
          </button>

          {sourcesOpen && (
            <div className="sources-list">
              {source_chunks.map((chunk, i) => {
                const fn = footnotesMap[chunk.chunk_id];
                const page = chunk.metadata?.page_number ?? chunk.page ?? "—";
                const docName = chunk.metadata?.source_doc ?? chunk.source_doc ?? "Document";
                return (
                  <div
                    key={chunk.chunk_id}
                    className="source-card"
                    onClick={() => handleSourceClick(chunk)}
                    title="Click to highlight in PDF"
                  >
                    <div className="source-card-header">
                      {fn && <span className="source-fn">{fn}</span>}
                      <span className="source-doc">{docName}</span>
                      <span className="source-page">p. {page}</span>
                    </div>
                    <p className="source-text">{chunk.text}</p>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
