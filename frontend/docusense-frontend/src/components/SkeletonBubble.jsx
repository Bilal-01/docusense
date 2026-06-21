import "./SkeletonBubble.css";

export default function SkeletonBubble() {
  return (
    <div className="skeleton-bubble">
      <div className="skeleton-badge" />
      <div className="skeleton-lines">
        <div className="skeleton-line" style={{ width: "92%" }} />
        <div className="skeleton-line" style={{ width: "78%" }} />
        <div className="skeleton-line" style={{ width: "85%" }} />
        <div className="skeleton-line" style={{ width: "55%" }} />
      </div>
      <div className="skeleton-sources" />
    </div>
  );
}
