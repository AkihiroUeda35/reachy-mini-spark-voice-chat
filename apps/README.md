# apps

Application entry points should live here.

Use the reusable STT, LLM, and TTS modules from [../lib](../lib) so app code stays focused on behavior and device orchestration.

If an app wants to honor a root `.env`, load it in the app entry point before importing modules that read environment-based defaults.

The next planned target is a Reachy Mini app that talks to the local STT, LLM, and TTS servers.