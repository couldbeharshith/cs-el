import { createSupabaseServer } from "../server";

export interface User {
  id: string;
  email: string;
  created_at: string;
}

export async function getUserByEmail(email: string): Promise<User | null> {
  const supabase = await createSupabaseServer();
  const { data, error } = await supabase
    .from("users")
    .select("*")
    .eq("email", email)
    .single();

  if (error) {
    // PGRST116 means no rows returned, which is not an error for this use case
    if (error.code === "PGRST116") return null;
    throw error;
  }
  return data as User;
}

export async function createUser(id: string, email: string): Promise<User> {
  const supabase = await createSupabaseServer();
  const { data, error } = await supabase
    .from("users")
    .insert({ id, email })
    .select()
    .single();

  if (error || !data) throw error ?? new Error("Failed to create user");
  return data as User;
}

