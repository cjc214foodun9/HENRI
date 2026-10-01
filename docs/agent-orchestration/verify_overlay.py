"""Read-only verification of the published HENRI instruction overlay.

This is a document/provenance check, not a HENRI model test.
"""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hermes-home',type=Path,help='Also compare exact installed overlay targets; never applies them.')
    args=parser.parse_args()
    base=Path(__file__).resolve().parent
    receipt=json.loads((base/'verification.json').read_text(encoding='utf-8'))
    overlay=base/'profile-overlay'
    errors=[]
    rows=receipt['overlay_files']
    if len({row['path'] for row in rows})!=len(rows):
        errors.append('duplicate manifest path')
    for row in rows:
        rel=Path(row['path'])
        if rel.is_absolute() or '..' in rel.parts:
            errors.append('unsafe manifest path: '+str(rel))
            continue
        p=overlay/rel
        if not p.is_file() or digest(p)!=row['sha256']:
            errors.append('overlay mismatch: '+str(rel))
        if args.hermes_home:
            p=args.hermes_home/rel
            if not p.is_file() or digest(p)!=row['sha256']:
                errors.append('installed mismatch: '+str(rel))
    declared={row['path'] for row in rows}
    actual={p.relative_to(overlay).as_posix() for p in overlay.rglob('*') if p.is_file()}
    if actual!=declared:
        errors.append('manifest/file-set mismatch')
    print(json.dumps({'scope':'instruction overlay bytes only; not model/CUDA performance',
        'status':'FAIL' if errors else 'PASS','checked_files':len(rows),
        'installed_comparison':bool(args.hermes_home),'errors':errors},indent=2))
    return 1 if errors else 0


if __name__=='__main__':
    raise SystemExit(main())
