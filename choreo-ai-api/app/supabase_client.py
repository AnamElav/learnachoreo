"""Supabase client helper for backend persistence."""

import os
from functools import lru_cache

from supabase import Client, create_client


@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    """Create a singleton Supabase admin client from env vars."""
    url = (os.getenv("SUPABASE_URL") or "").strip()
    service_role_key = (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
    if not url or not service_role_key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be configured")
    return create_client(url, service_role_key)

