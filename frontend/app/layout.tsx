import './globals.css'
import type { Metadata } from 'next'

import { ThemeProvider } from "@/components/global/theme-provider"
import QueryProvider from "@/components/global/query-provider";
import { Toaster } from "@/components/ui/sonner"


export const metadata: Metadata = {
  title: 'Sampada AI',
  description: 'Sampada AI - Financial Intelligence Assistant',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="h-full" suppressHydrationWarning>
      <body className="h-full font-sans">
          <QueryProvider>
            <ThemeProvider
              attribute="class"
              defaultTheme="system"
              enableSystem
              disableTransitionOnChange
            >
              {children}
            </ThemeProvider>
          </QueryProvider>
        <Toaster />
      </body>
    </html>
  )
}
