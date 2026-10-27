import { useState } from "react";
import { pct } from "../format";
import type { ModelResult } from "../types";

const LEGEND = [
  { cls: "equal", label: "Correct" },
  { cls: "substitute", label: "Substitution" },
  { cls: "insert", label: "Insertion" },
  { cls: "delete", label: "Deletion" },
];

export default function ErrorView({ results }: { results: ModelResult[] }) {
  const scored = results.filter((r) => r.alignment);
  const [picked, setPicked] = useState<string | null>(null);
  if (!scored.length) return null;

  const current = scored.find((r) => r.id === picked) ?? scored[0];
  const a = current.alignment!;

  return (
    <section className="card">
      <div className="row">
        <h2>Alignment &amp; error view</h2>
        <div className="seg-tabs" role="group" aria-label="Model to inspect">
          {scored.map((r) => (
            <button
              key={r.id}
              type="button"
              aria-pressed={r.id === current.id}
              onClick={() => setPicked(r.id)}
            >
              {r.label}
            </button>
          ))}
        </div>
      </div>
      <div className="km" lang="km">
        {a.segments.map((s, i) => (
          <span key={i} className={`seg ${s.type}`} title={s.type}>
            <span>{s.ref || " "}</span>
            <span>{s.hyp || " "}</span>
          </span>
        ))}
      </div>
      <div className="legend">
        {LEGEND.map((l) => (
          <span key={l.cls}><i className={l.cls} />{l.label}</span>
        ))}
        <span className="mono summary">
          S {a.substitutions} · I {a.insertions} · D {a.deletions} · CER {pct(a.cer)} of {a.reference_chars} chars
        </span>
      </div>
      <div className="note">
        Top: reference · bottom: model output. Characters are compared after normalization, with spaces removed.
      </div>
    </section>
  );
}
