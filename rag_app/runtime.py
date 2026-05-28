import asyncio
import os
import sys


if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def get_google_api_keys():
    keys = [
        os.getenv("GOOGLE_API_KEY"),
        os.getenv("GOOGLE_API_KEY_1"),
        os.getenv("GOOGLE_API_KEY_2"),
    ]
    return [key for key in keys if key]


def get_google_api_key():
    keys = get_google_api_keys()
    return keys[0] if keys else None


def close_google_model(model):
    client = getattr(model, "client", None)
    if client is None:
        return

    try:
        client.close()
    except Exception:
        pass

    try:
        async_client = getattr(client, "aio", None)
        if async_client is not None:
            async def close_async_client():
                await async_client.aclose()
                await asyncio.sleep(0.25)

            asyncio.run(close_async_client())
    except Exception:
        pass

    try:
        object.__setattr__(model, "client", None)
    except Exception:
        pass

