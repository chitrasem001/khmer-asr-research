// Shapes returned by app/server.py. Keep in sync with the FastAPI endpoints.

export interface ModelInfo {
  id: string;
  label: string;
  color: string;
  loaded: boolean;
  max_new_tokens: number | null;
}

export interface ModelsResponse {
  device: string;
  models: ModelInfo[];
}

export interface Sample {
  utt_id: string;
  speaker_id: string;
  duration: number;
  reference: string;
}

export type SegmentType = "equal" | "substitute" | "insert" | "delete";

export interface Segment {
  type: SegmentType;
  ref: string;
  hyp: string;
}

export interface Alignment {
  cer: number;
  substitutions: number;
  insertions: number;
  deletions: number;
  reference_chars: number;
  segments: Segment[];
}

export interface ModelResult {
  id: string;
  label: string;
  hypothesis: string;
  latency_ms: number;
  rtf: number;
  alignment?: Alignment;
}

export interface TranscribeResponse {
  duration: number;
  reference: string | null;
  device: string;
  results: ModelResult[];
}

export interface ResultRow {
  model: string;
  stage: string;
  clips: number;
  cer: number;
  cer_with_spaces: number;
  rtf: number;
  gpu: string | null;
  params_m: number;
  peak_vram_mb: number;
  fleurs_cer: number | null;
  fleurs_clips: number | null;
  sub: number;
  ins: number;
  del: number;
}

export type AudioSource =
  | { kind: "sample"; sample: Sample }
  | { kind: "file"; file: File };
