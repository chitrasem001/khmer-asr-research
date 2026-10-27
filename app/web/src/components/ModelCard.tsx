import { pct } from "../format";
import type { ModelResult } from "../types";

interface Props {
  result: ModelResult;
  color: string;
  maxNewTokens: number | null;
}

export default function ModelCard({ result, color, maxNewTokens }: Props) {
  const a = result.alignment;
  return (
    <article className="card" style={{ borderTop: `4px solid ${color}` }}>
      <div className="row">
        <h3>{result.label}</h3>
        {maxNewTokens && <span className="note">output capped at {maxNewTokens} tokens</span>}
      </div>
      <div className="hyp km" lang="km">
        {result.hypothesis || <span className="note">(empty output)</span>}
      </div>
      <div className="field">
        <div className="row" style={{ fontSize: 13 }}>
          <span className="note">CER</span>
          <span className="mono">{a ? pct(a.cer) : "no reference"}</span>
        </div>
        <div className="meter">
          <div style={{ width: `${a ? Math.min(100, 100 * a.cer) : 0}%`, background: color }} />
        </div>
      </div>
      <div className="stats">
        <div>
          <div>S / I / D</div>
          <div className="mono">{a ? `${a.substitutions} / ${a.insertions} / ${a.deletions}` : "—"}</div>
        </div>
        <div>
          <div>Latency</div>
          <div className="mono">{result.latency_ms.toLocaleString()} ms</div>
        </div>
        <div>
          <div>RTF</div>
          <div className="mono">{result.rtf}</div>
        </div>
      </div>
    </article>
  );
}
