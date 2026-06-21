import { useState, useRef, useEffect } from "react";
import MessageBubble from "./MessageBubble";
import SkeletonBubble from "./SkeletonBubble";
import "./ChatPanel.css";

const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

export default function ChatPanel({ docId, onHighlight }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const submit = async () => {
    const q = input.trim();
    if (!q || loading) return;

    setMessages((m) => [...m, { role: "user", text: q }]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch(`${API}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, doc_id: docId }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Query failed");
      }

      const data = await res.json();
      setMessages((m) => [...m, { role: "assistant", data }]);
    } catch (e) {
      setMessages((m) => [...m, { role: "error", text: e.message }]);
    } finally {
      setLoading(false);
    }
  };

  const onKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); }
  };

  return (
    <div className="chat-panel">
      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="chat-empty">
            <div className="chat-empty-icon">?</div>
            <p>Upload a document and ask anything about it.</p>
            <p className="chat-empty-sub">Answers will cite the exact source passage.</p>
          </div>
        )}

        {messages.map((msg, i) => {
          if (msg.role === "user") {
            return (
              <div key={i} className="bubble-user">
                <span>{msg.text}</span>
              </div>
            );
          }
          if (msg.role === "error") {
            return (
              <div key={i} className="bubble-error">
                {msg.text}
              </div>
            );
          }
          if (msg.role === "assistant") {
            return (
              <MessageBubble
                key={i}
                data={msg.data}
                onHighlight={onHighlight}
              />
            );
          }
          return null;
        })}

        {loading && <SkeletonBubble />}
        <div ref={bottomRef} />
      </div>

      <div className="chat-input-row">
        <textarea
          className="chat-input"
          placeholder={docId ? "Ask a question about the document…" : "Upload a document first…"}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKey}
          disabled={!docId || loading}
          rows={1}
        />
        <button
          className="chat-send"
          onClick={submit}
          disabled={!docId || loading || !input.trim()}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
            <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
          </svg>
        </button>
      </div>
    </div>
  );
}
