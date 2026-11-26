-- Run for a fresh, isolated Supabase project used by the shared demo workspace.
create table if not exists public.conversations (
  id uuid primary key default gen_random_uuid(),
  user_id text not null,
  title text not null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.messages (
  id text primary key default gen_random_uuid()::text,
  conversation_id uuid not null references public.conversations(id) on delete cascade,
  role text not null check (role in ('user', 'assistant')),
  content text not null,
  tool_invocations jsonb,
  created_at timestamptz not null default now()
);

create index if not exists conversations_user_updated_idx
  on public.conversations (user_id, updated_at desc);
create index if not exists messages_conversation_created_idx
  on public.messages (conversation_id, created_at);

alter table public.conversations enable row level security;
alter table public.messages enable row level security;
grant select, insert, update, delete on public.conversations to anon;
grant select, insert, update on public.messages to anon;

drop policy if exists demo_conversations on public.conversations;
create policy demo_conversations on public.conversations for all to anon
  using (user_id = 'demo-user') with check (user_id = 'demo-user');

drop policy if exists demo_messages on public.messages;
create policy demo_messages on public.messages for all to anon
  using (exists (
    select from public.conversations where id = conversation_id and user_id = 'demo-user'
  ))
  with check (exists (
    select from public.conversations where id = conversation_id and user_id = 'demo-user'
  ));

create or replace function public.touch_conversation_on_message()
returns trigger
language plpgsql
set search_path = ''
as $$
begin
  update public.conversations
  set updated_at = now(),
      title = case
        when title = 'New Chat' and new.role = 'user' then
          case when length(new.content) > 50 then left(new.content, 47) || '...' else new.content end
        else title
      end
  where id = new.conversation_id;
  return new;
end;
$$;

drop trigger if exists touch_conversation_on_message on public.messages;
create trigger touch_conversation_on_message after insert on public.messages
  for each row execute function public.touch_conversation_on_message();
