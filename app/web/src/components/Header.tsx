import type { ModelsResponse } from "../types";

export type Tab = "compare" | "results";

const TABS: { id: Tab; label: string }[] = [
  { id: "compare", label: "Compare" },
  { id: "results", label: "Results" },
];

interface Props {
  tab: Tab;
  onTab: (tab: Tab) => void;
  models: ModelsResponse | null;
}

export default function Header({ tab, onTab, models }: Props) {
  return (
    <header>
      <div className="wrap top">
        <div className="brand">
          <div className="logo">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
              <path d="M4 10v4M8 6v12M12 3v18M16 7v10M20 10v4" />
            </svg>
          </div>
          <div>
            <div className="title">Khmer ASR Studio</div>
            <div className="sub">Thesis demo · pretrained speech recognition for Khmer</div>
          </div>
        </div>
        <span className="pill">
          {models
            ? `${models.models.length} models · running on ${models.device.toUpperCase()}`
            : "Connecting…"}
        </span>
      </div>
      <nav aria-label="Sections">
        {TABS.map((t) => (
          <button
            key={t.id}
            type="button"
            aria-current={tab === t.id ? "page" : undefined}
            onClick={() => onTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>
    </header>
  );
}
