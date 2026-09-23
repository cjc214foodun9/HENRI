#!/usr/bin/env python3
"""STEP 3A: verify the GHCR image EXISTS before spending a create.

WHY NOT A TEMPLATE
    Template 725358 -- pinned by BOTH Vast skills -- is not in the live template
    list (2048 templates listed, 0 matches for "henri"). `vastai create instance`
    supports `--image`, and the repo's own workflow publishes the image:
        .github/workflows/docker-publish.yml
          REGISTRY=ghcr.io  IMAGE_NAME=${owner}/henri-v2-execution
          tags: branch | tag | sha-short | raw:latest (only on main)
    Owner comes from the git remote. So the image identity is REPO-DERIVED, not
    a guessed hash.
"""
from __future__ import annotations
import json, pathlib, subprocess, urllib.request, urllib.error

OWNER = "cjc214foodun9"
REPO  = "henri-v2-execution"

def sh(a, t=120):
    r = subprocess.run(a, capture_output=True, text=True, timeout=t)
    return r.returncode, r.stdout.strip(), r.stderr.strip()

# owner from the live remote (do not hardcode blind)
rc, out, _ = sh(["git", "remote", "get-url", "origin"])
if rc == 0 and "github.com" in out:
    OWNER = out.rstrip("/").split("github.com")[-1].strip(":/").split("/")[0]
print(f"owner_from_remote = {OWNER}")

# anonymous GHCR pull token
def token(owner, repo):
    url = (f"https://ghcr.io/token?scope=repository:{owner}/{repo}:pull"
           f"&service=ghcr.io")
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read()).get("token")

def manifest_ok(owner, repo, tag):
    try:
        tok = token(owner, repo)
    except Exception as e:
        return None, f"token_failed {e}"
    url = f"https://ghcr.io/v2/{owner}/{repo}/manifests/{tag}"
    req = urllib.request.Request(url, method="HEAD", headers={
        "Authorization": f"Bearer {tok}",
        "Accept": ("application/vnd.oci.image.index.v1+json,"
                   "application/vnd.docker.distribution.manifest.list.v2+json,"
                   "application/vnd.docker.distribution.manifest.v2+json"),
    })
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, "HTTPError"
    except Exception as e:
        return None, f"{type(e).__name__} {e}"

for tag in ("latest", "main"):
    code, info = manifest_ok(OWNER, REPO, tag)
    hdrs = info if isinstance(info, dict) else {}
    digest = hdrs.get("Docker-Content-Digest") or hdrs.get("docker-content-digest")
    print(f"GHCR {OWNER}/{REPO}:{tag} -> status={code} digest={digest}")

# local payload sizes (disk sizing)
TOP = pathlib.Path(r"C:\Users\chan\Desktop\HENRI 7B SWARM")
total = 0
for rel in ("environment_files", "HENRI V2/models"):
    p = TOP / rel
    if not p.exists():
        print(f"LOCAL {rel:22} MISSING")
        continue
    n = tot = 0
    for f in p.rglob("*"):
        if f.is_file():
            n += 1
            tot += f.stat().st_size
    total += tot
    print(f"LOCAL {rel:22} {n:6d} files {tot/1048576:9.1f} MiB")
print(f"LOCAL payload_total = {total/1048576:.1f} MiB")
