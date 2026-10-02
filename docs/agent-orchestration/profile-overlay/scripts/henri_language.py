"""HENRI-STE-V1 advisory language checker. Not full ASD-STE100 validation.

Operational prose has soft targets. Formal and visual text stays exact.
This module never rewrites text, calls models, or permits execution.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re

VERSION = 'HENRI-STE-V1'
SCAFFOLD = (
    'HENRI-STE-V1: Use short, active sentences for operational prose. '
    'Target 20 words per instruction and 25 words per description. '
    'Use one task per sentence. Use one term per meaning. '
    'Keep code, equations, identifiers, hashes, paths, and source quotes exact. '
    'Avoid progressive and perfect verbs in operational prose. '
    'Do not use passive procedures. Use WARNING for injury and CAUTION for damage. '
    'Use editable diagrams and accessible HTML for substantive user reports and human decisions. '
    'Visual prose and layout remain unrestricted. Keep safety text and a text alternative. '
    'Style findings are advice, not proof, permission, or task success. '
    'Keep the existing prompt prefix fixed. Append task state last.'
)
CHOICES = {'ensure': 'make sure', 'prior to': 'before', 'commence': 'start',
           'commencing': 'start', 'utilize': 'use', 'replenish': 'fill', 'in order to': 'to'}
PROGRESSIVE = re.compile(r'\b(?:am|is|are|was|were|be|been)\s+(?:\w+ly\s+)?(?:being\s+)?\w+ing\b', re.I)
PARTICIPLE = r'(?:\w+ed|written|done|made|taken|seen|gone|given|known|run|built|read|sent|set|put|cut|lost|kept|found)'
PERFECT = re.compile(r'\b(?:has|have|had)\s+(?:not\s+)?' + PARTICIPLE + r'\b', re.I)
PASSIVE = re.compile(r'\b(?:must\s+be|should\s+be|is\s+to\s+be)\s+(?:not\s+)?' + PARTICIPLE + r'\b', re.I)
PASSIVE_CANDIDATE = re.compile(r'\b(?:am|is|are|was|were|be|been)\s+(?:not\s+)?' + PARTICIPLE + r'\b', re.I)
ING_ADJECTIVES = {'missing', 'remaining', 'interesting'}


def digest(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def assess(text: str, *, track: str, mode: str = 'description', hazard: str | None = None) -> dict:
    if not isinstance(text, str) or len(text.encode('utf-8')) > 65536 or '\x00' in text:
        raise ValueError('bounded UTF-8 text required')
    if track not in {'operational', 'formal', 'visual'}:
        raise ValueError('explicit supported track required')
    if mode not in {'procedure', 'description'} or hazard not in {None, 'injury', 'damage'}:
        raise ValueError('mode or hazard invalid')
    findings = []
    protected = 0
    sentences = []
    if track == 'operational':
        # Local heuristic count: not the official Rule 8 counter or a grammar AST.
        # Never mask unclosed fences: malformed markup must not hide later prose.
        fence = False
        lines = []
        for line in text.splitlines():
            if line.lstrip().startswith('```'):
                fence = not fence
                protected += 1
            elif fence or line.lstrip().startswith('>'):
                protected += 1
            else:
                lines.append(re.sub(r'`[^`\n]+`', ' IDENTIFIER ', line))
        if fence:
            raise ValueError('unclosed code fence')
        prose = '\n'.join(lines)
        if prose.count('`') % 2:
            findings.append({'rule': 'unclosed_identifier', 'review': 'The output may be truncated. Do not adopt it as an instruction.'})
        # Decimal points remain exact; sentence boundaries require terminal punctuation.
        sentences = [s.strip() for s in re.split(r'(?<!\d)[.!?](?!\d)(?:\s+|$)|\n+', prose) if s.strip()]
        limit = 20 if mode == 'procedure' else 25
        for i, sentence in enumerate(sentences, 1):
            words = re.findall(r"\b[\w]+(?:[-'][\w]+)*\b", sentence)
            if len(words) > limit:
                findings.append({'rule': 'length_target', 'sentence': i, 'words': len(words), 'target': limit})
            for term, alternative in CHOICES.items():
                if re.search(r'\b' + re.escape(term) + r'\b', sentence, re.I):
                    findings.append({'rule': 'word_choice', 'sentence': i, 'term': term, 'suggestion': alternative})
            for rule, pattern in [('progressive', PROGRESSIVE), ('perfect', PERFECT)]:
                match=pattern.search(sentence)
                if match and not (rule=='progressive' and match.group(0).split()[-1].lower() in ING_ADJECTIVES):
                    findings.append({'rule': rule, 'sentence': i, 'review': 'Confirm the verb form before any edit.'})
            if mode == 'procedure' and PASSIVE.search(sentence):
                findings.append({'rule': 'passive_procedure', 'sentence': i, 'review': 'State the actor or use a command.'})
            elif mode == 'procedure' and PASSIVE_CANDIDATE.search(sentence):
                findings.append({'rule': 'passive_candidate', 'sentence': i, 'review': 'This can be an adjective or a passive verb. Review the meaning before an edit.'})
        paragraphs = [p for p in re.split(r'\n\s*\n', prose) if p.strip()]
        for i, paragraph in enumerate(paragraphs, 1):
            count = len([s for s in re.split(r'(?<!\d)[.!?](?!\d)(?:\s+|$)', paragraph) if s.strip()])
            if count > 6:
                findings.append({'rule': 'paragraph_target', 'paragraph': i, 'sentences': count, 'target': 6})
    # Safety labels are independent of the style exemption. Hazard must be explicit.
    expected = {'injury': 'WARNING:', 'damage': 'CAUTION:'}.get(hazard)
    if expected and not text.lstrip().startswith(expected):
        findings.append({'rule': 'safety_marker', 'expected': expected, 'review': 'Use the correct marker and state the risk.'})
    return {'version': VERSION, 'track': track, 'mode': mode,
            'status': 'ADVISORY' if findings else ('EXEMPT' if track != 'operational' else 'NO_SELECTED_FINDINGS'),
            'findings': findings, 'sentences_assessed': len(sentences), 'protected_lines': protected,
            'input_sha256': digest(text), 'output_sha256': digest(text), 'rewritten': False,
            'authorization': False, 'full_ste_compliance': False,
            'coverage': 'selected patterns and heuristic counts; not full dictionary, syntax, meaning, or noun groups'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file', type=Path, required=True)
    parser.add_argument('--track', choices=['operational', 'formal', 'visual'], required=True)
    parser.add_argument('--mode', choices=['procedure', 'description'], default='description')
    parser.add_argument('--hazard', choices=['injury', 'damage'])
    args = parser.parse_args()
    try:
        if args.file.stat().st_size > 65536:
            raise ValueError('input size bound exceeded')
        result = assess(args.file.read_text(encoding='utf-8'), track=args.track, mode=args.mode, hazard=args.hazard)
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0
    except (OSError, ValueError, TypeError) as exc:
        print(json.dumps({'version': VERSION, 'status': 'BLOCKED', 'authorization': False, 'error_type': type(exc).__name__}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
