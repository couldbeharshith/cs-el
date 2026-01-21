"use client"

import { useEffect } from "react"
import { useRouter } from "next/navigation"

export default function LandingPage() {
  const router = useRouter()
  
  useEffect(() => {
    // Redirect directly to chat
    router.push("/chat")
  }, [router])
  
  return null
}