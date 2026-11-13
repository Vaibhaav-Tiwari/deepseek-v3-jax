"""Download the public-domain crime corpus, clean it, train a BPE tokenizer, write token files.

Run:  .venv/bin/python prepare_data.py
Out:  data/train.bin, data/val.bin (uint16 token ids), data/tokenizer.json
"""
import io, re, sys, random, tarfile
from html.parser import HTMLParser
from pathlib import Path
import numpy as np
import requests
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

DATA, RAW = Path("data"), Path("data/raw")
VOCAB, DOC = 8192, "<|doc|>"          # <|doc|> separates books, token id 0
HAMMETT_REPEAT = 3                     # upsample the hardboiled core in the train split
START = re.compile(r"^\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*$", re.M | re.I)
END = re.compile(r"^\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG.*$", re.M | re.I)
CREDITS = re.compile(r"^(End of (the |this )?Project Gutenberg|Produced by|Scanned by|E-?text prepared|Transcribed by|This e-?book|Transcriber)|Distributed Proofread|pgdp\.net|Internet Archive", re.I)
TOC_ENTRY = re.compile(r"^(CHAPTER\s+)?([IVXLC]+|\d+)[.:]?\s|^[^a-z]+$")
QUOTES = str.maketrans({"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'"})
SKIP_XHTML = {"colophon", "imprint", "titlepage", "halftitlepage", "uncopyright", "toc", "loi", "endnotes", "dedication", "epigraph"}


def strip_gutenberg(text: str, name: str) -> str:
    """Keep only the book between Project Gutenberg's START/END markers."""
    s, e = START.search(text), END.search(text)
    if not s or not e:
        raise ValueError(f"{name}: Project Gutenberg START/END markers not found")
    return text[s.end():e.start()]


class _Paragraphs(HTMLParser):
    def __init__(self):
        super().__init__(); self.paras, self.buf, self.depth = [], [], 0
    def handle_starttag(self, tag, attrs):
        if tag in ("p", "h2", "h3", "blockquote"): self.depth += 1
    def handle_endtag(self, tag):
        if tag in ("p", "h2", "h3", "blockquote") and self.depth:
            self.depth -= 1
            if not self.depth and "".join(self.buf).strip():
                self.paras.append(" ".join("".join(self.buf).split())); self.buf = []
    def handle_data(self, data):
        if self.depth: self.buf.append(data)


def strip_xhtml(html: str) -> str:
    """Standard Ebooks chapter XHTML -> paragraphs separated by blank lines."""
    p = _Paragraphs(); p.feed(html)
    return "\n\n".join(p.paras).translate(QUOTES)


def clean_text(text: str) -> str:
    """Unwrap hard-wrapped paragraphs; drop credits, the table of contents, illustration tags,
    _italic_ underscores; straighten curly quotes."""
    text = re.sub(r"\[(Illustration|Transcriber|Frontispiece)[^\]]*\]|Ver\.[0-9.]+\*END\*", "", text.replace("\r\n", "\n"))
    paras = [" ".join(p.split()) for p in re.split(r"\n\s*\n", text)]
    out, in_toc = [], False
    for p in paras:
        if not p or CREDITS.search(p): continue
        if p.upper() == "CONTENTS": in_toc = True; continue
        if in_toc and TOC_ENTRY.search(p): continue
        in_toc = False; out.append(p)
    return "\n\n".join(out).replace("_", "").translate(QUOTES)


def split_docs(docs: list[str], val_frac: float = 0.02, seed: int = 0) -> tuple[list[str], list[str]]:
    """Deterministic split by whole document, at least one document in val."""
    idx = list(range(len(docs))); random.Random(seed).shuffle(idx)
    n_val = max(1, round(len(docs) * val_frac))
    return [docs[i] for i in sorted(idx[n_val:])], [docs[i] for i in sorted(idx[:n_val])]


def fetch(url: str) -> bytes:
    path = RAW / re.sub(r"[^A-Za-z0-9._-]", "_", url.split("://")[1])
    if not path.exists():
        r = requests.get(url, timeout=60); r.raise_for_status(); path.write_bytes(r.content)
    return path.read_bytes()


def standard_ebook(tgz: bytes) -> str:
    """Chapters of a Standard Ebooks source repo tarball, in reading (spine) order."""
    tar = tarfile.open(fileobj=io.BytesIO(tgz))
    files = {Path(m.name).name: m for m in tar.getmembers() if "/src/epub/" in m.name and m.isfile()}
    opf = tar.extractfile(files["content.opf"]).read().decode()
    items = [dict(re.findall(r'(\w+)="([^"]*)"', tag)) for tag in re.findall(r"<item\s[^>]*>", opf)]
    hrefs = {i["id"]: i["href"][len("text/"):] for i in items if i.get("href", "").startswith("text/")}
    spine = [hrefs[i] for i in re.findall(r'<itemref[^>]*idref="([^"]+)"', opf) if i in hrefs]
    body = [f for f in spine if Path(f).stem not in SKIP_XHTML]
    return "\n\n".join(strip_xhtml(tar.extractfile(files[f]).read().decode()) for f in body)


def load_sources(path="sources.txt") -> list[tuple[str, str, str]]:
    lines = [l.split("\t") for l in Path(path).read_text().splitlines() if l and not l.startswith("#")]
    return [(kind, url, title) for kind, url, title, *_ in lines]


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    hammett, pulp = [], []
    for i, (kind, url, title) in enumerate(load_sources()):
        print(f"[{i + 1:3d}] {title[:70]}", file=sys.stderr)
        if kind == "standardebooks":
            hammett.append(standard_ebook(fetch(url)))
        else:
            pulp.append(clean_text(strip_gutenberg(fetch(url).decode("utf-8-sig", errors="replace"), title)))
    train, val = split_docs(pulp)                 # Hammett always trains; val is held-out crime fiction
    train = train + hammett * HAMMETT_REPEAT

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(train, trainers.BpeTrainer(vocab_size=VOCAB, special_tokens=[DOC],
                                                       initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok.save(str(DATA / "tokenizer.json"))

    for name, docs in [("train", train), ("val", val)]:
        ids = [[0] + e.ids for e in tok.encode_batch(docs)]
        arr = np.concatenate(ids).astype(np.uint16)
        assert arr.max() < 65536 and tok.token_to_id(DOC) == 0
        arr.tofile(DATA / f"{name}.bin")
        print(f"{name}: {len(docs)} docs, {sum(map(len, docs)) / 1e6:.1f}M chars, {len(arr) / 1e6:.2f}M tokens")


if __name__ == "__main__":
    main()
