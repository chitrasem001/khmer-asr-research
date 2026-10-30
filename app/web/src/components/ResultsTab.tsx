import { useEffect, useState } from "react";
import { getResults } from "../api";
import { pct } from "../format";
import type { ResultRow } from "../types";

export default function ResultsTab() {
  const [rows, setRows] = useState<ResultRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getResults()
      .then((r) => setRows(r.rows))
      .catch((err: Error) => setError(err.message));
  }, []);

  return (
    <section className="card">
      <h2>Model summary · test set (unseen speakers and sentences)</h2>
      {error && <div className="error">{error}</div>}
      <div className="scroll-x">
        <table aria-label="Model summary">
          <thead>
            <tr>
              <th>Model</th>
              <th>Stage</th>
              <th className="num">Clips</th>
              <th className="num">CER</th>
              <th className="num">CER w/ spaces</th>
              <th className="num">FLEURS CER</th>
              <th className="num">RTF</th>
              <th>GPU</th>
              <th className="num">Params</th>
              <th className="num">Peak VRAM</th>
            </tr>
          </thead>
          <tbody>
            {rows?.map((r) => (
              <tr key={`${r.stage}-${r.model}`}>
                <td>{r.model}</td>
                <td>{r.stage}</td>
                <td className="num">{r.clips.toLocaleString()}</td>
                <td className="num"><b>{pct(r.cer)}</b></td>
                <td className="num">{pct(r.cer_with_spaces)}</td>
                <td className="num">{r.fleurs_cer == null ? "pending" : pct(r.fleurs_cer)}</td>
                <td className="num">{r.rtf.toFixed(3)}</td>
                <td>{r.gpu?.replace(/^NVIDIA (GeForce )?/, "") ?? "—"}</td>
                <td className="num">{Math.round(r.params_m)} M</td>
                <td className="num">{(r.peak_vram_mb / 1024).toFixed(1)} GB</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="note">
        Read live from <code>experiments/m0_baseline</code>, <code>m1_eval</code> and their FLEURS folders.
        Zero-shot Whisper is scored on a seeded 100-clip-per-speaker sample. RTF depends on the GPU, so
        compare it only between rows on the same GPU.
      </div>
    </section>
  );
}
