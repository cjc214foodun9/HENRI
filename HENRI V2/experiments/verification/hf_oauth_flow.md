# HuggingFace OAuth — working flow for this machine

The `hf auth login` device flow races a 300 s browser round-trip and has timed
out repeatedly. The token-paste flow below is one step and does not race the
clock. Tokens are NEVER pasted into chat content; they are stored locally via
the CLI and referenced by label only.

## Preferred: token paste (one-time, ~30 s)

1. Browser -> https://huggingface.co/settings/tokens
2. New token; name = `henri-orchestrator`; role = Read; Generate.
3. Copy the token (starts with `hf_`). It appears once.
4. Paste into the terminal (the agent can stage it with the CLI flag):

```bash
hf auth login --token "$HF_TOKEN" --force
hf auth whoami
```

Do NOT paste the token into chat content/markdown. Put it in the terminal
prompt if the agent opens one, or set it in the environment.

## Device flow (when preferred)

1. The agent mints the code: `hf auth login --force` (background, pty).
2. Within 300 s, open https://hf.co/oauth/device and enter the code shown.
3. The CLI completes and stores the token in
   `C:\Users\chan\.cache\huggingface\token`.

## Gated dataset `cais/hle` (HLE)

After login is complete, in a browser open
https://huggingface.co/datasets/cais/hle and click **Agree and access
repository**. Acceptance is account-side; the CLI token alone does not
auto-accept gated=auto datasets. Verify after:

```bash
huggingface_hub.HfApi().dataset_info("cais/hle")  # shows gated=auto, private=False
```

## Recovery

- `Error: Login failed: Device code expired (timeout)` -> mint a new code
  (reply "new code") and authorize within 5 minutes.
- Invalid stored token -> `hf auth login --force` or the token-paste flow.

## Revocation

https://huggingface.co/settings/tokens -> Revoke.
