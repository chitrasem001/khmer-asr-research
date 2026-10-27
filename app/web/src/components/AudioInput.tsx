import { useEffect, useMemo } from "react";
import { audioUrl } from "../api";
import type { AudioSource, Sample } from "../types";

interface Props {
  samples: Sample[];
  source: AudioSource | null;
  onSource: (source: AudioSource) => void;
}

export default function AudioInput({ samples, source, onSource }: Props) {
  // Object URLs for uploads must be released when the file changes.
  const fileUrl = useMemo(
    () => (source?.kind === "file" ? URL.createObjectURL(source.file) : null),
    [source],
  );
  useEffect(() => () => { if (fileUrl) URL.revokeObjectURL(fileUrl); }, [fileUrl]);

  const src = source?.kind === "sample" ? audioUrl(source.sample.utt_id) : fileUrl ?? undefined;
  const sampleIndex = source?.kind === "sample" ? samples.indexOf(source.sample) : -1;

  return (
    <div className="card">
      <div className="row">
        <h2>Audio input</h2>
        <label className="btn">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d="M12 16V4M7 9l5-5 5 5M4 20h16" />
          </svg>
          Upload
          <input
            type="file"
            accept="audio/*"
            className="visually-hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) onSource({ kind: "file", file });
              e.target.value = "";
            }}
          />
        </label>
      </div>
      <audio controls src={src} style={{ width: "100%" }} />
      <div className="field">
        <label className="small" htmlFor="sample-select">Load a test-set sample</label>
        <select
          id="sample-select"
          value={sampleIndex >= 0 ? String(sampleIndex) : ""}
          onChange={(e) => onSource({ kind: "sample", sample: samples[Number(e.target.value)] })}
        >
          {sampleIndex < 0 && <option value="">— uploaded file —</option>}
          {samples.map((s, i) => (
            <option key={s.utt_id} value={i}>
              {s.utt_id} · speaker {s.speaker_id} · {s.duration.toFixed(1)} s
            </option>
          ))}
        </select>
      </div>
      <div className="note">
        {source?.kind === "sample"
          ? `Speaker ${source.sample.speaker_id} was never seen in training, and neither was this sentence.`
          : "WAV, FLAC or OGG up to 60 s; resampled to 16 kHz like the test set."}
      </div>
    </div>
  );
}
