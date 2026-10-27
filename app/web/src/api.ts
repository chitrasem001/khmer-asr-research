import type { ModelsResponse, ResultRow, Sample, TranscribeResponse } from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? res.statusText);
  }
  return res.json() as Promise<T>;
}

export const getModels = () => request<ModelsResponse>("/api/models");
export const getSamples = () => request<Sample[]>("/api/samples");
export const getResults = () => request<{ rows: ResultRow[] }>("/api/results");
export const audioUrl = (uttId: string) => `/api/audio/${encodeURIComponent(uttId)}`;

export function transcribe(params: {
  models: string[];
  uttId?: string;
  file?: File;
  reference?: string;
}): Promise<TranscribeResponse> {
  const form = new FormData();
  form.append("models", params.models.join(","));
  if (params.uttId) form.append("utt_id", params.uttId);
  if (params.file) form.append("file", params.file);
  if (params.reference) form.append("reference", params.reference);
  return request<TranscribeResponse>("/api/transcribe", { method: "POST", body: form });
}
