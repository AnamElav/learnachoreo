"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { supabase } from "@/lib/supabase/client";

export default function LoginPage() {
  const router = useRouter();
  const searchParams = useSearchParams();

  const next = useMemo(() => searchParams.get("next") ?? "/", [searchParams]);

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const [loading, setLoading] = useState(false);
  const [oauthLoading, setOauthLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function bootstrapSession() {
      try {
        const { data } = await supabase.auth.getSession();
        if (cancelled) return;
        if (data.session?.user) {
          const target = next && next.startsWith("/") ? next : "/";
          router.replace(target);
        }
      } catch {
        // ignore
      }
    }

    void bootstrapSession();

    return () => {
      cancelled = true;
    };
  }, [next, router]);

  const onEmailLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const { error: err } = await supabase.auth.signInWithPassword({
        email,
        password,
      });
      if (err) throw err;
      const target = next && next.startsWith("/") ? next : "/";
      router.replace(target);
      router.refresh();
    } catch (err2) {
      const msg = err2 instanceof Error ? err2.message : "Login failed";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const onGoogleLogin = async () => {
    setError(null);
    setOauthLoading(true);
    try {
      const redirectTo = `${window.location.origin}/auth/login`;
      const { error: err } = await supabase.auth.signInWithOAuth({
        provider: "google",
        options: { redirectTo },
      });
      if (err) throw err;
    } catch (err2) {
      const msg = err2 instanceof Error ? err2.message : "OAuth failed";
      setError(msg);
      setOauthLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0d0d0d] text-zinc-100 flex flex-col items-center justify-center px-4 font-sans">
      <div className="w-full max-w-md mx-auto space-y-6">
        <header className="text-center">
          <h1 className="text-2xl font-semibold tracking-tight text-white">
            Sign in
          </h1>
          <p className="mt-1 text-sm text-zinc-500">
            Welcome back. Continue your practice with Choreo AI.
          </p>
        </header>

        <div className="rounded-xl bg-zinc-900/80 border border-zinc-800 p-6 space-y-4">
          <form onSubmit={onEmailLogin} className="space-y-4">
            <label className="block">
              <span className="text-sm text-zinc-500">Email</span>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                className="mt-1 w-full h-11 px-4 rounded-xl bg-zinc-900 border border-zinc-800 text-white placeholder:text-zinc-500 focus:outline-none focus:ring-2 focus:ring-emerald-500/50 focus:border-emerald-500 transition"
                disabled={loading || oauthLoading}
              />
            </label>

            <label className="block">
              <span className="text-sm text-zinc-500">Password</span>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                className="mt-1 w-full h-11 px-4 rounded-xl bg-zinc-900 border border-zinc-800 text-white placeholder:text-zinc-500 focus:outline-none focus:ring-2 focus:ring-emerald-500/50 focus:border-emerald-500 transition"
                disabled={loading || oauthLoading}
              />
            </label>

            <button
              type="submit"
              disabled={loading || oauthLoading}
              className="w-full h-12 rounded-xl bg-emerald-600 text-white font-medium hover:bg-emerald-500 disabled:opacity-50 disabled:cursor-not-allowed transition"
            >
              {loading ? "Signing in..." : "Sign in"}
            </button>
          </form>

          <div className="relative">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-zinc-800" />
            </div>
            <div className="relative flex justify-center text-xs">
              <span className="px-3 bg-zinc-900/80 text-zinc-500">or</span>
            </div>
          </div>

          <button
            type="button"
            onClick={onGoogleLogin}
            disabled={loading || oauthLoading}
            className="w-full h-12 rounded-xl bg-white/5 border border-zinc-700 text-zinc-100 font-medium hover:bg-white/10 disabled:opacity-50 disabled:cursor-not-allowed transition"
          >
            {oauthLoading ? "Opening Google..." : "Continue with Google"}
          </button>

          {error && (
            <p className="text-sm text-red-400 leading-relaxed bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">
              {error}
            </p>
          )}

          <p className="text-sm text-zinc-400">
            New here?{" "}
            <button
              type="button"
              className="text-emerald-400 hover:text-emerald-300 font-medium"
              onClick={() => router.push("/auth/signup")}
              disabled={loading || oauthLoading}
            >
              Create an account
            </button>
          </p>
        </div>
      </div>
    </div>
  );
}

