import { cookies } from "next/headers";
import { createServerClient } from "@supabase/ssr";

/**
 * Server-side Supabase client configured to read/write the user session cookie.
 * Use in server code (middleware, route handlers, server actions).
 */
export function createSupabaseServerClient() {
  // Next's `cookies()` type can vary by Next.js version/context.
  // We cast to avoid TS mismatch; runtime still uses the cookie store.
  type CookieStoreLike = {
    getAll: () => Array<{ name: string; value: string }>;
    set: (cookie: { name: string; value: string; [key: string]: unknown }) => void;
  };

  const cookieStore = cookies() as unknown as CookieStoreLike;

  return createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL ?? "",
    process.env.SUPABASE_SERVICE_ROLE_KEY ?? "",
    {
      cookies: {
        getAll() {
          return cookieStore.getAll();
        },
        setAll(cookiesToSet, _headers) {
          void _headers;
          // Next's cookie store expects an object: { name, value, ...options }.
          // In some contexts this may be read-only; we safely no-op in that case.
          try {
            cookiesToSet.forEach(({ name, value, options }) => {
              cookieStore.set({
                name,
                value,
                ...((options ?? {}) as Record<string, unknown>),
              });
            });
          } catch {
            // no-op
          }
        },
      },
    }
  );
}

