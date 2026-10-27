import { useEffect, useState } from "react";
import { getSamples, transcribe } from "../api";
import { FALLBACK_COLOR } from "../format";
import type { AudioSource, ModelInfo, Sample, TranscribeResponse } from "../types";
import AudioInput from "./AudioInput";
import ErrorView from "./ErrorView";
import ModelCard from "./ModelCard";

export default function CompareTab({ models }: { models: ModelInfo[] }) {
  const [samples, setSamples] = useState<Sample[]>([]);
  const [source, setSource] = useState<AudioSource | null>(null);
  const [typedReference, setTypedReference] = useState("");
  const [selected, setSelected] = useState<string[]>(models.map((m) => m.id));
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [output, setOutput] = useState<TranscribeResponse | null>(null);

  useEffect(() => {
    getSamples()
      .then((s) => {
        setSamples(s);
        if (s.length) setSource({ kind: "sample", sample: s[0] });
      })
      .catch((err: Error) => setError(err.message));
  }, []);

  const pickSource = (next: AudioSource) => {
    setSource(next);
    setTypedReference("");
    setOutput(null);
  };

  const toggle = (id: string) =>
    setSelected((cur) => (cur.includes(id) ? cur.filter((x) => x !== id) : [...cur, id]));

  const run = async () => {
    if (!source) return;
    if (!selected.length) return setError("Pick at least one model.");
    setBusy(true);
    setError(null);
    setStatus("Transcribing… the first run loads each model, which can take a minute.");
    try {
      // Keep the models in config order regardless of click order.
      const ids = models.map((m) => m.id).filter((id) => selected.includes(id));
      const out = await transcribe(
        source.kind === "sample"
          ? { models: ids, uttId: source.sample.utt_id }
          : { models: ids, file: source.file, reference: typedReference },
      );
      setOutput(out);
      setStatus(`Done · clip ${out.duration} s on ${out.device.toUpperCase()}`);
    } catch (err) {
      setError((err as Error).message);
      setStatus("");
    } finally {
      setBusy(false);
    }
  };

  const model = (id: string) => models.find((m) => m.id === id);

  return (
    <section className="stack">
      <div className="grid2">
        <AudioInput samples={samples} source={source} onSource={pickSource} />
        <div className="card">
          <div className="row">
            <h2>Reference transcript</h2>
            <span className="tag">
              {source?.kind === "sample" ? `test set · ${source.sample.utt_id}` : source ? `upload · ${source.file.name}` : ""}
            </span>
          </div>
          {source?.kind === "file" ? (
            <textarea
              className="ref km"
              lang="km"
              rows={3}
              value={typedReference}
              onChange={(e) => setTypedReference(e.target.value)}
              placeholder="Optional: type the correct Khmer text to score the uploaded clip"
            />
          ) : (
            <div className="ref km" lang="km">{source?.sample.reference}</div>
          )}
          <div className="note">CER uses the thesis normalization and ignores spaces, exactly as in the paper.</div>
        </div>
      </div>

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
          <div className="row">
            <span className="note" role="status">{status}</span>
            <button type="button" className="btn primary" onClick={run} disabled={busy || !source}>
              {busy ? "Transcribing…" : "Transcribe"}
            </button>
          </div>
        </div>
      </div>

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
          <ErrorView results={output.results} />
        </>
      )}
    </section>
  );
}
