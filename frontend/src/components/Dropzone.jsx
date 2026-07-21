import { useCallback, useRef, useState } from "react";

export default function Dropzone({ label, file, previewUrl, onFile }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  const handleFiles = useCallback(
    (files) => {
      if (files && files[0]) onFile(files[0]);
    },
    [onFile]
  );

  return (
    <div
      className="dropzone"
      data-dragging={dragging}
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        handleFiles(e.dataTransfer.files);
      }}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => e.key === "Enter" && inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        hidden
        onChange={(e) => handleFiles(e.target.files)}
      />
      {previewUrl ? (
        <img src={previewUrl} alt={label} className="dropzone-preview" />
      ) : (
        <div className="dropzone-empty">
          <span className="dropzone-icon">⤒</span>
          <span>{label}</span>
          <span className="dropzone-hint">drop an image or click to browse</span>
        </div>
      )}
      {file && (
        <div className="dropzone-filename mono" title={file.name}>
          {file.name}
        </div>
      )}
    </div>
  );
}
