"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Share, Plus, X, Download } from "lucide-react"

export function InstallPrompt() {
  const [isIOS, setIsIOS] = useState(false)
  const [isStandalone, setIsStandalone] = useState(false)
  const [deferredPrompt, setDeferredPrompt] = useState<any>(null)
  const [isVisible, setIsVisible] = useState(false)

  useEffect(() => {
    const isIOSDevice = /iPad|iPhone|iPod/.test(navigator.userAgent) && !(window as any).MSStream
    setIsIOS(isIOSDevice)

    const isStandaloneMode = window.matchMedia('(display-mode: standalone)').matches
    setIsStandalone(isStandaloneMode)
    
    if (isStandaloneMode) return;

    // Show for iOS immediately if not standalone
    if (isIOSDevice) {
        setIsVisible(true)
    }

    const handleBeforeInstallPrompt = (e: any) => {
      e.preventDefault()
      setDeferredPrompt(e)
      setIsVisible(true)
    }

    window.addEventListener('beforeinstallprompt', handleBeforeInstallPrompt)

    return () => {
      window.removeEventListener('beforeinstallprompt', handleBeforeInstallPrompt)
    }
  }, [])

  const handleInstallClick = async () => {
    if (!deferredPrompt) return

    deferredPrompt.prompt()

    const { outcome } = await deferredPrompt.userChoice
    
    if (outcome === 'accepted') {
      setDeferredPrompt(null)
      setIsVisible(false)
    }
  }

  if (isStandalone) {
    return null
  }

  if (!isVisible) {
      return null
  }

  return (
    <div className="fixed bottom-4 left-4 right-4 z-50 md:left-auto md:right-4 md:w-96">
      <Card className="relative shadow-lg border-primary/20 bg-background/95 backdrop-blur">
        <Button 
            variant="ghost" 
            size="icon" 
            className="absolute right-2 top-2 h-6 w-6"
            onClick={() => setIsVisible(false)}
        >
            <X className="h-4 w-4" />
        </Button>
        <CardHeader className="pb-2">
          <CardTitle className="text-lg flex items-center gap-2">
            <Download className="h-5 w-5 text-primary" />
            Install App
          </CardTitle>
          <CardDescription>
            Add this app to your home screen for a better experience.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {!isIOS && (
             <Button onClick={handleInstallClick} className="w-full">
                Add to Home Screen
             </Button>
          )}
          {isIOS && (
            <div className="text-sm text-muted-foreground">
              <p className="mb-2">
                To install this app on your iOS device:
              </p>
              <ol className="list-decimal list-inside space-y-1 ml-1">
                  <li>Tap the share button <Share className="inline h-4 w-4 mx-1" /></li>
                  <li>Scroll down and tap "Add to Home Screen" <Plus className="inline h-4 w-4 mx-1" /></li>
              </ol>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
