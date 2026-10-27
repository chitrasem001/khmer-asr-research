import { useEffect, useState } from "react";
import { getModels } from "./api";
import Header, { type Tab } from "./components/Header";
import CompareTab from "./components/CompareTab";
import ResultsTab from "./components/ResultsTab";
import type { ModelsResponse } from "./types";

const tabFromHash = (): Tab => (window.location.hash === "#results" ? "results" : "compare");

export default function App() {
  const [tab, setTabState] = useState<Tab>(tabFromHash);
  // The URL hash keeps the open tab, so a link can open the Results tab directly.
  const setTab = (next: Tab) => {
    window.location.hash = next;
    setTabState(next);
  };
  useEffect(() => {
    const onHash = () => setTabState(tabFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  const [models, setModels] = useState<ModelsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getModels()
      .then(setModels)
      .catch((err: Error) => setError(`Could not reach the server: ${err.message}`));
  }, []);

  return (
    <>
      <Header tab={tab} onTab={setTab} models={models} />
      <main>
        {error && <div className="error">{error}</div>}
        {models && (tab === "compare" ? <CompareTab models={models.models} /> : <ResultsTab />)}
      </main>
    </>
  );
}
