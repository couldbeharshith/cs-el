'use server'

export async function subscribeUser(subscription: PushSubscriptionJSON) {
  // Store the subscription in your database
  // For now, just log it - implement actual storage as needed
  console.log('User subscribed:', subscription.endpoint)
  return { success: true }
}

export async function unsubscribeUser(endpoint: string) {
  // Remove the subscription from your database
  // For now, just log it - implement actual removal as needed
  console.log('User unsubscribed:', endpoint)
  return { success: true }
}

