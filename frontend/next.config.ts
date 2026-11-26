import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  devIndicators: false,
  outputFileTracingIncludes: {
    "/api/chat": ["./features/chat/lib/ai/prompts/*.md"],
  },
  images: {
    remotePatterns: [{ hostname: "*", protocol: "https" }],
  },
};

export default nextConfig;
