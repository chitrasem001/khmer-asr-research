export const pct = (x: number | null | undefined) =>
  x == null ? "—" : `${(100 * x).toFixed(1)}%`;

export const FALLBACK_COLOR = "#3A4250";
