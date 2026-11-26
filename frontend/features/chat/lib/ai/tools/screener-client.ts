const SCREENER_API_URL = process.env.SCREENER_API_URL || "http://127.0.0.1:8080";

export function screenerUrl(path: string) {
  return new URL(SCREENER_API_URL.replace(/\/$/, "") + path);
}

export function integrationSignal(signal: AbortSignal | undefined) {
  const timeout = AbortSignal.timeout(25_000);
  return signal ? AbortSignal.any([signal, timeout]) : timeout;
}
