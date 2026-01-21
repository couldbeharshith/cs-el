
const TABLE_SCHEMA = `

// Example table schema for a "users" table, replace with your actual schema

CREATE TABLE users (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

`;

export async function getTableSchema(): Promise<string> {
  return TABLE_SCHEMA;
}
