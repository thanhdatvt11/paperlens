"""Find main sections by matching numbered TOC titles to original body text.

This deliberately does not infer a three-level tree or use a model to guess
chapter boundaries. Quotes and line spans always refer to immutable source.
"""
from dataclasses import dataclass
import re
import unicodedata

from app.services import document_structure_service as engine


@dataclass(frozen=True)
class TocEntry:
    number: int
    title: str
    chunk_id: int
    page_number: int
    start_line: int
    end_line: int
    quote: str


_NUMBERED = re.compile(r'^\s*(?:#{1,6}\s*)?(\d{1,3})[.)](?!\d)\s*(\S.*)$')
_BULLET = re.compile(r'^\s*[•¢*«●▪◦\-]\s*')


def _clean_entry(text):
    return re.sub(r'\s*\.{2,}\s*\d+\s*$', '', text).strip()


def _key(text):
    """Comparison only; never rewrite displayed quotes or source titles."""
    text = unicodedata.normalize('NFKC',text)
    text = re.sub(r'^\s*#{1,6}\s*','',text)
    text = _NUMBERED.sub(lambda m:m[2],text,count=1)
    # Parenthetical subtitles can wrap differently in the PDF/OCR.
    text = text.split('(')[0]
    text = text.casefold()
    # Common scanned AI/Al distinction; exact quote is still preserved.
    text = re.sub(r'\bal\b','ai',text)
    return ' '.join(re.findall(r'[^\W_]+',text,flags=re.UNICODE))


def _toc_pages(chunks):
    return {c.page_number for c in chunks if any(
        re.fullmatch(r'(?:table of )?contents|mục lục',line.strip().lstrip('# '),re.I)
        for line in c.text.splitlines()[:8])}


def read_toc_entries(chunks):
    pages = _toc_pages(chunks)
    if not pages:
        raise engine.StructureOutputError('No explicit table of contents found; main-section matching requires a numbered TOC. No guessed structure was produced.')
    entries=[]
    warnings=[]
    for c in chunks:
        if c.page_number not in pages:
            continue
        lines=c.text.splitlines(keepends=True)
        for pos,raw in enumerate(lines):
            match=_NUMBERED.match(raw.strip())
            if not match:
                continue
            number=int(match[1])
            title=_clean_entry(match[2])
            end=pos+1
            # Only a directly adjacent continuation can extend a TOC title.
            for j in range(pos+1,min(len(lines),pos+3)):
                text=lines[j].strip()
                if (not text or _NUMBERED.match(text) or _BULLET.match(text)
                        or text.isdigit() or 'copyright' in text.casefold()):
                    break
                title+=' '+_clean_entry(text)
                end=j+1
            if not _key(title):
                continue
            entries.append(TocEntry(number,title,c.chunk_id,c.page_number,pos+1,end,
                                    ''.join(lines[pos:end]).strip()))
    if len(entries)<2:
        raise engine.StructureOutputError('Fewer than two numbered TOC entries found; inspect extracted TOC before summaries.')
    numbers=[e.number for e in entries]
    if len(set(numbers))!=len(numbers) or numbers!=sorted(numbers):
        raise engine.StructureOutputError('Ambiguous or repeated TOC numbering; automatic main sections were not accepted.')
    if numbers!=list(range(numbers[0],numbers[-1]+1)):
        warnings.append('TOC numbering has gaps; missing entries require inspection.')
    return entries,pages,warnings


def _match_score(entry,text):
    left,right=_key(entry.title),_key(text)
    if not right:
        return 0
    number=_NUMBERED.match(text.strip().lstrip('# '))
    if number and int(number[1])!=entry.number:
        return 0
    if left==right:
        return 1.0
    # Truncated source heading allowed only for a long literal prefix, never
    # token similarity or matching on a chapter number alone.
    words=right.split()
    if len(words)>=6 and left.startswith(right+' ') and len(right)/len(left)>=.82:
        return .9
    return 0


def build_main_sections(chunks):
    chunks=sorted(chunks,key=lambda c:(c.chunk_index,c.chunk_id))
    if not chunks or any(not c.text.strip() for c in chunks):
        raise engine.StructureOutputError('Main sections require nonempty snapshots')
    if len({c.chunk_id for c in chunks})!=len(chunks) or len({c.chunk_index for c in chunks})!=len(chunks):
        raise engine.StructureOutputError('Duplicate chunk IDs/indices in structure input')
    if any(b.page_number<a.page_number for a,b in zip(chunks,chunks[1:])):
        raise engine.StructureOutputError('Chunk page order is inconsistent')
    if any(b.chunk_index!=a.chunk_index+1 for a,b in zip(chunks,chunks[1:])):
        raise engine.StructureOutputError('Main-section probe needs contiguous chunks; omit --pages')
    chunks=engine.prepare_structure_chunks(chunks)
    entries,toc_pages,warnings=read_toc_entries(chunks)
    toc_last=max(toc_pages)
    candidates=[]
    for ci,c in enumerate(chunks):
        if c.page_number<=toc_last:
            continue
        lines=c.text.splitlines(keepends=True)
        code_lines=engine._code_lines(c.text)
        for begin,raw in enumerate(lines,1):
            if (not raw.strip() or begin in code_lines
                    or (not any(ch.isalpha() for ch in raw) and not re.fullmatch(r'\d+[.)]',raw.strip()))):
                continue
            for end in range(begin,min(len(lines),begin+2)+1):
                if (not lines[end-1].strip() or not any(ch.isalpha() for ch in lines[end-1]) or any(i in code_lines for i in range(begin,end+1))):
                    continue
                quote=''.join(lines[begin-1:end]).strip()
                if len(quote)>240:
                    continue
                candidates.append((ci,begin,end,quote))
    matches=[]
    previous=(-1,0)
    for entry in entries:
        eligible=[]
        for ci,begin,end,quote in candidates:
            if (ci,begin)<=previous:
                continue
            score=_match_score(entry,quote)
            if score:
                eligible.append((score,ci,begin,end,quote))
        if not eligible:
            warnings.append(f'TOC entry {entry.number} not matched in body: {entry.title}. No chapter boundary invented; surrounding text is retained.')
            continue
        # First occurrence in source order; shortest highest-scoring quote at it.
        eligible.sort(key=lambda x:(x[1],x[2],-x[0],x[3]-x[2]))
        score,ci,begin,end,quote=eligible[0]
        matches.append((entry,ci,begin,end,quote,score))
        previous=(ci,end)
        if score<1:
            warnings.append(f'TOC entry {entry.number} matched a truncated source prefix; inspect heading quote.')
    if len(matches)<2:
        raise engine.StructureOutputError('Fewer than two main headings matched the TOC; inspect source instead of generating a misleading summary.')
    selected={}
    for entry,ci,begin,end,quote,score in matches:
        selected.setdefault(chunks[ci].chunk_id,[]).append(engine._Heading(start_line=begin,end_line=end,level=1))
        warnings.append(f'TOC {entry.number} -> chunk {chunks[ci].chunk_id}, page {chunks[ci].page_number}, lines {begin}-{end}; TOC source chunk {entry.chunk_id}, lines {entry.start_line}-{entry.end_line}.')
    windows=[]
    for c in chunks:
        kind='table_of_contents' if c.page_number in toc_pages else engine._source_kind(c)
        windows.append(engine._Window(c,1,len(c.text.splitlines(keepends=True)),
                       engine._Analysis(kind=kind,headings=selected.get(c.chunk_id,[]))))
    sections=engine._build_sections(chunks,windows)
    unmatched=[e for e in entries if e.number not in {m[0].number for m in matches}]
    warnings[:0]=[
        'Strategy: numbered TOC title matching; only main sections are produced, not a three-level tree.',
        'Main-section detection uses server source matching, not Ollama. Section/global summaries still use the configured Ollama adapter.',
        'All snapshot lines remain assigned exactly once; quote identity is not semantic verification.',
        'Source before the first matched main section is retained under a generated label.',
        'Content absent from the TOC stays inside surrounding matched sections; this is not proof of a complete editorial outline.',
        'Body boundaries are textual matches; OCR variants, repeated headings, unnumbered appendices and TOC errors require review.',
        f'TOC entries: {len(entries)}; matched: {len(matches)}; unmatched: {len(unmatched)}.',
    ]
    # Các phần chính đều cùng cấp.
    for section in sections:
        section.parent_id = None
        section.level = 1

    return sections, warnings
