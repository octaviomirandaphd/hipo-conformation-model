"""
The figure code exists in two places: src/hipo/_figblock.py (imported by
src/hipo/plots.py) and, verbatim, inside hipo_model_standalone.py.  Two copies of
anything is how the original four analysis scripts drifted apart, so this test
asserts they are byte-identical.

    cd src && PYTHONPATH=. python ../tests/test_figure_parity.py
"""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
block = (root / "src" / "hipo" / "_figblock.py").read_text(encoding="utf-8")
standalone = (root / "hipo_model_standalone.py").read_text(encoding="utf-8")

BEGIN = "# --- BEGIN _figblock.py ---------------------------------------------------\n"
END = "# --- END _figblock.py -----------------------------------------------------\n"

assert BEGIN in standalone and END in standalone, \
    "standalone is missing the _figblock delimiters"
embedded = standalone.split(BEGIN, 1)[1].split(END, 1)[0]

if embedded != block:
    import difflib
    d = list(difflib.unified_diff(block.splitlines(), embedded.splitlines(),
                                  "_figblock.py", "standalone", lineterm="", n=1))
    print("\n".join(d[:60]))
    raise SystemExit("**FAIL** figure code has drifted between the two copies")

print(f"PASS  figure block identical in both copies ({len(block.splitlines())} lines)")
