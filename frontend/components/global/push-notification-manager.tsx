'use client'

import { useState, useEffect } from 'react'
import { subscribeUser, unsubscribeUser } from '@/actions/notifications'
import { Button } from '@/components/ui/button'
import { Bell, BellOff } from 'lucide-react'
import { toast } from 'sonner'

function urlBase64ToUint8Array(base64String: string) {
  const padding = '='.repeat((4 - (base64String.length % 4)) % 4)
  const base64 = (base64String + padding)
    .replace(/-/g, '+')
    .replace(/_/g, '/')

  const rawData = window.atob(base64)
  const outputArray = new Uint8Array(rawData.length)

  for (let i = 0; i < rawData.length; ++i) {
    outputArray[i] = rawData.charCodeAt(i)
  }
  return outputArray
}

export function PushNotificationManager() {
  const [isSupported, setIsSupported] = useState(false)
  const [subscription, setSubscription] = useState<PushSubscription | null>(null)

  useEffect(() => {
    if ('serviceWorker' in navigator && 'PushManager' in window) {
      setIsSupported(true)
      registerServiceWorker()
    }
  }, [])

  async function registerServiceWorker() {
    try {
      const registration = await navigator.serviceWorker.register('/sw.js', {
        scope: '/',
        updateViaCache: 'none',
      })
      const sub = await registration.pushManager.getSubscription()
      setSubscription(sub)
    } catch (error) {
      console.error('Service Worker registration failed:', error)
    }
  }

  async function subscribeToPush() {
    try {
      const registration = await navigator.serviceWorker.ready
      const sub = await registration.pushManager.subscribe({
        userVisibleOnly: true,
        applicationServerKey: urlBase64ToUint8Array(
          process.env.NEXT_PUBLIC_VAPID_PUBLIC_KEY!
        ),
      })
      setSubscription(sub)
      const serializedSub = JSON.parse(JSON.stringify(sub))
      await subscribeUser(serializedSub)
      toast.success('Subscribed to notifications!')
    } catch (error) {
      console.error('Failed to subscribe', error)
      toast.error('Failed to subscribe to notifications')
    }
  }

  async function unsubscribeFromPush() {
    try {
        if (subscription) {
            await subscription.unsubscribe()
            await unsubscribeUser(subscription.endpoint)
            setSubscription(null)
            toast.success('Unsubscribed from notifications')
        }
    } catch (error) {
        console.error('Failed to unsubscribe', error)
        toast.error('Failed to unsubscribe')
    }
  }

  if (!isSupported) {
    return null
  }

  return (
    <div className="flex items-center gap-2">
      {subscription ? (
        <Button variant="outline" size="sm" onClick={unsubscribeFromPush}>
          <BellOff className="mr-2 h-4 w-4" />
          Disable Notifications
        </Button>
      ) : (
        <Button variant="outline" size="sm" onClick={subscribeToPush}>
          <Bell className="mr-2 h-4 w-4" />
          Enable Notifications
        </Button>
      )}
    </div>
  )
}
