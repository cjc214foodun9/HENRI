"""Offline Git-pinned engineering memory. Not an ontology or audit ledger."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

REPO_URL = 'https://github.com/cjc214foodun9/HENRI.git'
CLASSES = {'OBSERVED', 'DERIVED', 'INFERRED', 'HYPOTHESIS', 'FALSIFIED', 'BLOCKED'}


def strict_json(text: str | bytes) -> object:
    def pairs(rows):
        value = {}
        for key, item in rows:
            if key in value:
                raise ValueError('duplicate JSON key')
            value[key] = item
        return value
    def nonfinite(_value):
        raise ValueError('nonfinite JSON value')
    return json.loads(text, object_pairs_hook=pairs, parse_constant=nonfinite)


def check_store_path(store: Path) -> None:
    for component in (store, *store.parents):
        try:
            stat = component.lstat()
        except FileNotFoundError:
            continue
        if component.is_symlink() or getattr(stat, 'st_file_attributes', 0) & 0x400:
            raise ValueError('memory store ancestor link or junction refused')


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')


def git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(['git', '-C', str(repo), *args], stderr=subprocess.PIPE, timeout=30)


def commit(repo: Path, value: str) -> str:
    if not re.fullmatch('[0-9a-f]{40}', value):
        raise ValueError('full lowercase commit SHA required')
    if git(repo, 'cat-file', '-t', value).strip() != b'commit':
        raise ValueError('source object is not a commit')
    return value


def safe_source(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or '\\' in value or ':' in value or '..' in path.parts or str(path) != value:
        raise ValueError('source path traversal or noncanonical path')
    allowed = value.startswith('HENRI V2/agentic_graph/') or value.startswith('docs/agent-orchestration/')
    if not allowed or path.suffix not in {'.py', '.md', '.yaml', '.json', '.html', '.drawio'}:
        raise ValueError('source outside approved engineering scope')
    if any(part.lower() in {'secrets', 'dataset', 'data', 'memories', 'logs', '.env'} for part in path.parts):
        raise ValueError('sensitive source path')
    return value


def source_ref(repo: Path, sha: str, value: str) -> dict:
    name = safe_source(value)
    blob = git(repo, 'rev-parse', sha + ':' + name).decode().strip()
    if git(repo, 'cat-file', '-t', blob).strip() != b'blob':
        raise ValueError('source is not a blob')
    data = git(repo, 'cat-file', 'blob', blob)
    return {'path': name, 'git_blob': blob, 'sha256': hashlib.sha256(data).hexdigest()}


def public_text(value: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum or any(ord(c) < 32 and c not in '\n\t' for c in value):
        raise ValueError('bounded nonempty public text required')
    sensitive = r'(?i)(?:password|[a-z_ -]*token|api[_ -]?key|secret)\s*[:=]\s*[^\s]{4,}|-----BEGIN.*PRIVATE KEY|sk-[A-Za-z0-9_-]{12,}|Bearer\s+[A-Za-z0-9._-]{8,}|AKIA[0-9A-Z]{16}'
    excluded = r'(?i)gold[_ -]?answers?|benchmark[_ -]?answers?|answer[_ -]?keys?|ground[_ -]?truth|held[_ -]?out[_ -]?answers?|raw[_ -]?transcript|zone[_ -]?c[_ -]?payload|latent[_ -]?tensor'
    if re.search(sensitive, value) or re.search(excluded, value):
        raise ValueError('sensitive or benchmark/latent content refused')
    return value


def record_id(body: dict) -> str:
    return 'pm-' + hashlib.sha256(canonical(body)).hexdigest()


def validate(repo: Path, record: dict) -> None:
    keys = {'schema_version', 'repo_url', 'source_commit', 'title', 'summary', 'evidence_class', 'sources', 'record_id'}
    if not isinstance(record, dict) or set(record) != keys or record['schema_version'] != 1:
        raise ValueError('record schema invalid')
    if record['repo_url'] != REPO_URL or record['evidence_class'] not in CLASSES:
        raise ValueError('record scope or evidence class invalid')
    public_text(record['title'], 160)
    public_text(record['summary'], 2000)
    sha = commit(repo, record['source_commit'])
    body = {k: v for k, v in record.items() if k != 'record_id'}
    if record['record_id'] != record_id(body):
        raise ValueError('record content hash drift')
    if not isinstance(record['sources'], list) or not 1 <= len(record['sources']) <= 12:
        raise ValueError('one to twelve source refs required')
    for ref in record['sources']:
        if not isinstance(ref, dict) or set(ref) != {'path', 'git_blob', 'sha256'}:
            raise ValueError('source ref schema invalid')
        if ref != source_ref(repo, sha, ref['path']):
            raise ValueError('source hash drift')


def read_records(repo: Path, store: Path) -> list[dict]:
    check_store_path(store)
    if not store.is_dir():
        raise ValueError('memory store missing')
    if store.is_symlink():
        raise ValueError('memory store symlink refused')
    records = []
    paths = sorted(store.glob('*.json'))
    if len(paths) > 1000:
        raise ValueError('record count bound exceeded')
    for path in paths:
        if path.is_symlink():
            raise ValueError('memory record symlink refused')
        if path.stat().st_size > 16384:
            raise ValueError('record size bound exceeded')
        record = strict_json(path.read_text(encoding='utf-8'))
        validate(repo, record)
        if path.name != record['record_id'] + '.json':
            raise ValueError('record filename identity drift')
        records.append(record)
    if not records:
        raise ValueError('memory store empty')
    return records


def add(repo: Path, store: Path, args: argparse.Namespace) -> dict:
    if not args.reviewed_public:
        raise ValueError('explicit public review acknowledgement required')
    sha = commit(repo, args.commit)
    body = {'schema_version': 1, 'repo_url': REPO_URL, 'source_commit': sha,
            'title': args.title, 'summary': args.summary, 'evidence_class': args.evidence_class,
            'sources': [source_ref(repo, sha, name) for name in sorted(set(args.source))]}
    record = dict(body, record_id=record_id(body))
    validate(repo, record)
    check_store_path(store)
    store.mkdir(parents=True, exist_ok=True)
    if store.is_symlink():
        raise ValueError('memory store symlink refused')
    path = store / (record['record_id'] + '.json')
    try:
        with path.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(record, sort_keys=True, indent=2, ensure_ascii=False) + '\n')
    except FileExistsError:
        if path.is_symlink() or strict_json(path.read_text(encoding='utf-8')) != record:
            raise ValueError('existing record identity conflict')
    return {'status': 'LOCAL_RECORDED', 'record_id': record['record_id'], 'path': str(path),
            'scope': 'public operational record; not signed truth or governance'}


def remote_verify(repo: Path, store: Path, branch: str) -> dict:
    if store.resolve() != (repo / 'docs/project-memory/records').resolve():
        raise ValueError('remote verification requires default repository record store')
    if not isinstance(branch, str) or not branch or branch.startswith('-'):
        raise ValueError('exact branch name required')
    git(repo, 'check-ref-format', 'refs/heads/' + branch)
    if git(repo, 'remote', 'get-url', 'origin').decode().strip() != REPO_URL:
        raise ValueError('remote outside approved repository')
    sha = git(repo, 'rev-parse', 'HEAD').decode().strip()
    rows = git(repo, 'ls-remote', 'origin', 'refs/heads/' + branch).decode().split()
    if len(rows) != 2 or rows[0] != sha:
        raise ValueError('remote branch SHA differs from local HEAD')
    records = read_records(repo, store)
    for record in records:
        name = 'docs/project-memory/records/' + record['record_id'] + '.json'
        published = strict_json(git(repo, 'show', sha + ':' + name))
        if published != record:
            raise ValueError('local record differs from published commit')
    return {'status': 'REMOTE_VERIFIED', 'repo_url': REPO_URL, 'branch': branch,
            'local_sha': sha, 'remote_sha': rows[0], 'verified_records': len(records),
            'scope': 'Git ref and committed record objects; not CI/main or Honcho'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['add', 'verify', 'query', 'remote-verify', 'honcho-sync'])
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--store', type=Path)
    parser.add_argument('--commit')
    parser.add_argument('--source', action='append', default=[])
    parser.add_argument('--title', default='')
    parser.add_argument('--summary', default='')
    parser.add_argument('--evidence-class', choices=sorted(CLASSES), default='INFERRED')
    parser.add_argument('--query', default='')
    parser.add_argument('--limit', type=int, default=8)
    parser.add_argument('--branch')
    parser.add_argument('--reviewed-public', action='store_true', help='Acknowledge review of public title, summary, and source refs; not an automated secrecy proof.')
    args = parser.parse_args()
    try:
        repo = args.repo.resolve()
        store = args.store or repo / 'docs/project-memory/records'
        if args.action == 'add':
            result = add(repo, store, args)
        elif args.action == 'verify':
            result = {'status': 'PASS', 'checked_records': len(read_records(repo, store))}
        elif args.action == 'query':
            sha = commit(repo, args.commit)
            if not 1 <= args.limit <= 20:
                raise ValueError('limit must be 1..20')
            terms = public_text(args.query, 240).casefold().split()
            all_records = read_records(repo, store)
            if not any(r['source_commit'] == sha for r in all_records):
                raise ValueError('no verified record for requested source SHA; history is not current recall')
            records = [r for r in all_records if r['source_commit'] == sha
                       and all(t in (r['title'] + ' ' + r['summary']).casefold() for t in terms)]
            selected = records[:args.limit]
            result = {'status': 'LOCAL_RECALL', 'source_commit': sha, 'count': len(selected),
                      'matched_total': len(records), 'records': selected, 'provider_cache_hit': False}
        elif args.action == 'honcho-sync':
            raise RuntimeError('Honcho projection is offline by operator policy')
        elif args.action == 'remote-verify':
            result = remote_verify(repo, store, args.branch)
        else:
            raise ValueError('unsupported action')
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError, KeyError, TypeError) as exc:
        print(json.dumps({'status': 'BLOCKED', 'error_type': type(exc).__name__, 'error': str(exc)[:350]}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
