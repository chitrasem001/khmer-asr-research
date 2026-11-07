import { useEffect, useMemo, useRef, useState } from "react";
import { transcribe } from "../api";
import { FALLBACK_COLOR } from "../format";
import { Recorder } from "../recorder";
import type { ModelInfo, TranscribeResponse } from "../types";
import ErrorView from "./ErrorView";
import ModelCard from "./ModelCard";

const MAX_SECONDS = 60; // the server accepts clips up to 60 s
const UPDATE_MS = 1500; // how often live captions are refreshed
const PREFERRED_LIVE = "mms-m2-100h"; // CTC decoding is fast enough for live captions

export default function LiveTab({ models }: { models: ModelInfo[] }) {
  const recorder = useRef<Recorder | null>(null);
  const inFlight = useRef(false);
  const [recording, setRecording] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [level, setLevel] = useState(0);
  const [liveModel, setLiveModel] = useState(
    models.some((m) => m.id === PREFERRED_LIVE) ? PREFERRED_LIVE : models[0]?.id ?? "",
  );
  const [caption, setCaption] = useState("");
  const [clip, setClip] = useState<File | null>(null);
  const [reference, setReference] = useState("");
  const [selected, setSelected] = useState<string[]>(models.map((m) => m.id));
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [output, setOutput] = useState<TranscribeResponse | null>(null);

  const clipUrl = useMemo(() => (clip ? URL.createObjectURL(clip) : null), [clip]);
  useEffect(() => () => { if (clipUrl) URL.revokeObjectURL(clipUrl); }, [clipUrl]);

  // Refresh the live caption while recording: send everything captured so far to one model.
  useEffect(() => {
    if (!recording) return;
    const timer = window.setInterval(async () => {
      const rec = recorder.current;
      if (!rec) return;
      setSeconds(rec.seconds);
      if (rec.seconds >= MAX_SECONDS) {
        void stop();
        return;
      }
      if (inFlight.current || rec.seconds < 0.8) return; // one request at a time
      inFlight.current = true;
      try {
        const out = await transcribe({ models: [liveModel], file: rec.wav() });
        if (recorder.current === rec) setCaption(out.results[0]?.hypothesis ?? "");
      } catch (err) {
        setError((err as Error).message);
      } finally {
        inFlight.current = false;
      }
    }, UPDATE_MS);
    return () => window.clearInterval(timer);
  }, [recording, liveModel]);

  const start = async () => {
    setError(null);
    setOutput(null);
    setClip(null);
    setCaption("");
    setSeconds(0);
    const rec = new Recorder();
    try {
      await rec.start(setLevel);
    } catch (err) {
      const e = err as Error;
      setError(
        e.name === "NotAllowedError"
          ? "Microphone access was blocked. Allow it in the browser's address bar and try again."
          : `Could not start the microphone: ${e.message}. Open the demo at http://localhost:8000.`,
      );
      return;
    }
    recorder.current = rec;
    setRecording(true);
    setStatus("Listening… speak Khmer. The caption updates every 1.5 s.");
  };

  const stop = async () => {
    const rec = recorder.current;
    if (!rec) return;
    recorder.current = null;
    setRecording(false);
    setLevel(0);
    const file = rec.wav(`live-${new Date().toISOString().slice(11, 19).replace(/:/g, "")}.wav`);
    await rec.stop();
    setSeconds(rec.seconds);
    setClip(file);
    setStatus(`Recorded ${rec.seconds.toFixed(1)} s. Compare all models on it below.`);
    // Final caption from the live model on the complete recording.
    try {
      const out = await transcribe({ models: [liveModel], file });
      setCaption(out.results[0]?.hypothesis ?? "");
    } catch (err) {
      setError((err as Error).message);
    }
  };

  const toggle = (id: string) =>
    setSelected((cur) => (cur.includes(id) ? cur.filter((x) => x !== id) : [...cur, id]));

  const compare = async () => {
    if (!clip) return;
    if (!selected.length) return setError("Pick at least one model.");
    setBusy(true);
    setError(null);
    setStatus("Transcribing… the first run loads each model, which can take a minute.");
    try {
      const ids = models.map((m) => m.id).filter((id) => selected.includes(id));
      const out = await transcribe({ models: ids, file: clip, reference });
      setOutput(out);
      setStatus(`Done · ${out.duration} s on ${out.device.toUpperCase()}`);
    } catch (err) {
      setError((err as Error).message);
      setStatus("");
    } finally {
      setBusy(false);
    }
  };

  const model = (id: string) => models.find((m) => m.id === id);
  const liveColor = model(liveModel)?.color ?? FALLBACK_COLOR;

  return (
    <section className="stack">
      <div className="card">
        <div className="row">
          <h2>Live microphone</h2>
          <label className="row" style={{ gap: 8 }}>
            <span className="small">Live caption model</span>
            <select
              value={liveModel}
              onChange={(e) => setLiveModel(e.target.value)}
              disabled={recording}
              style={{ width: "auto" }}
            >
              {models.map((m) => (
                <option key={m.id} value={m.id}>{m.label}</option>
              ))}
            </select>
          </label>
        </div>

        <div className="live-controls">
          <button
            type="button"
            className={`mic ${recording ? "on" : ""}`}
            onClick={recording ? stop : start}
            aria-pressed={recording}
            aria-label={recording ? "Stop recording" : "Start recording"}
          >
            {recording ? (
              <svg width="28" height="28" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                <rect x="6" y="6" width="12" height="12" rx="2" />
              </svg>
            ) : (
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
                <rect x="9" y="3" width="6" height="11" rx="3" />
                <path d="M5 11a7 7 0 0 0 14 0M12 18v3" />
              </svg>
            )}
          </button>
          <div className="field" style={{ flex: 1 }}>
            <div className="row" style={{ fontSize: 13 }}>
              <span className="note">{recording ? "Recording" : clip ? "Stopped" : "Press the microphone to start"}</span>
              <span className="mono">{seconds.toFixed(1)} / {MAX_SECONDS} s</span>
            </div>
            <div className="meter" aria-hidden="true">
              <div style={{ width: `${Math.round(level * 100)}%`, background: liveColor, transition: "width 80ms" }} />
            </div>
          </div>
        </div>

        <div className="ref km live-caption" lang="km" aria-live="polite" style={{ borderColor: liveColor }}>
          {caption || <span className="note">The transcript appears here while you speak.</span>}
        </div>
        <div className="note" role="status">{status}</div>
      </div>

      {clip && clipUrl && (
        <div className="grid2">
          <div className="card">
            <h2>Your recording</h2>
            <audio controls src={clipUrl} style={{ width: "100%" }} />
            <a className="note" href={clipUrl} download={clip.name}>Download the WAV file</a>
          </div>
          <div className="card">
            <h2>What you said (optional)</h2>
            <textarea
              className="ref km"
              lang="km"
              rows={3}
              value={reference}
              onChange={(e) => setReference(e.target.value)}
              placeholder="Type the Khmer sentence you said to score each model's errors"
            />
            <div className="note">CER uses the thesis normalization and ignores spaces.</div>
          </div>
        </div>
      )}

      {clip && (
        <div className="card compact">
          <div className="row">
            <fieldset>
              <legend>Models</legend>
              {models.map((m) => (
                <label key={m.id} className="toggle">
                  <input
                    type="checkbox"
                    checked={selected.includes(m.id)}
                    onChange={() => toggle(m.id)}
                    style={{ accentColor: m.color }}
                  />
                  <span className="dot" style={{ background: m.color }} />
                  {m.label}
                </label>
              ))}
            </fieldset>
            <button type="button" className="btn primary" onClick={compare} disabled={busy}>
              {busy ? "Transcribing…" : "Compare models"}
            </button>
          </div>
        </div>
      )}

      {error && <div className="error">{error}</div>}

      {output && (
        <>
          <div className="cards">
            {output.results.map((r) => (
              <ModelCard
                key={r.id}
                result={r}
                color={model(r.id)?.color ?? FALLBACK_COLOR}
                maxNewTokens={model(r.id)?.max_new_tokens ?? null}
              />
            ))}
          </div>
          {output.reference && <ErrorView results={output.results} />}
        </>
      )}
    </section>
  );
}
