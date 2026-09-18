#!/usr/bin/env python3
"""Role-identity audit: verify attestation receipts against evidence written by a
different process than the one being audited.

Why this signal and not the detector's. The detector asks "does an episodic record
exist, dated near this signature?". Retrospectively that is weak: the records are
untracked (0 of 195 in git), their only timestamps are a filename date and a mutable
mtime, and they sit inside the Director's own write scope. A signature carries a date
and nothing finer, so same-day remediation verifies.

The runner's session logs are different evidence. run_agi_lab.sh writes them, not the
Director's tool calls; tools/stream_formatter.py prints the AGENT line straight from the
Task call's `subagent_type` parameter; and they carry wall-clock timestamps. So for every
episodic receipt filed under a role, we can ask a question the Director cannot answer for
itself: was the *registered agent of that role* ever actually instantiated?

Two verdicts are provable without trusting any Director-written file:
  NO_ROLE_DISPATCH        the role was never dispatched by subagent_type, anywhere
  PRECEDES_FIRST_DISPATCH the receipt is dated before that role's first real dispatch
And one structural flag, which is how the D-109 remediation slipped through:
  GENERIC_SUBSTITUTION    subagent_type=general-purpose, description names a role
"""
import re, csv, json, datetime as dt
from collections import defaultdict, Counter
from pathlib import Path

REPO = Path.home() / "mnt/AGI"
LOGS = REPO / "data/infra/session_logs"
OUT  = Path.home() / "census"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
HDR  = re.compile(r"=== Session (\d+) starting at (.+?) ===")
AGENT = re.compile(r">>> AGENT\s+([A-Za-z0-9_\-]+)\s*(?:—\s*(.*))?")
LAUNCH = re.compile(r">>> Launching agent:\s*([A-Za-z0-9_\-]+)")
MONTHS = {m: i+1 for i, m in enumerate("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split())}
GENERIC = {"general-purpose", "general", "superpowers"}

roles = sorted(json.loads((REPO/"data/agents/agents.json").read_text()).keys())
role_re = {r: re.compile(r"\b" + re.escape(r).replace("_", "[_ ]") + r"\b", re.I) for r in roles}

def hdr_time(s):
    m = re.match(r"\w+ (\w+) +(\d+) (\d+):(\d+):(\d+) \S+ (\d+)", s.strip())
    if not m or m.group(1) not in MONTHS: return None
    mo, d, H, M, S, Y = m.groups()
    return dt.datetime(int(Y), MONTHS[mo], int(d), int(H), int(M), int(S))

# ---- 1. dispatch timeline, straight from the runner's own output ----
dispatches, per_role, generic_sub = [], defaultdict(list), []
logs = sorted(LOGS.glob("*.log"))
for lf in logs:
    txt = ANSI.sub("", lf.read_text(errors="replace"))
    m = HDR.search(txt)
    t = hdr_time(m.group(2)) if m else None
    if t is None:
        fm = re.search(r"_(\d{8})_(\d{6})\.log$", lf.name)
        t = dt.datetime.strptime(fm.group(1)+fm.group(2), "%Y%m%d%H%M%S") if fm else None
    for line in txt.splitlines():
        am = AGENT.search(line)
        if am:
            st, desc = am.group(1), (am.group(2) or "").strip()
            dispatches.append((t, st, desc, lf.name))
            per_role[st].append(t)
            if st in GENERIC:
                named = [r for r in roles if r != "director" and role_re[r].search(desc)]
                if named:
                    generic_sub.append({"when": t, "log": lf.name, "subagent_type": st,
                                        "named_roles": "|".join(named), "desc": desc[:110]})
            continue
        lm = LAUNCH.search(line)
        if lm:
            dispatches.append((t, lm.group(1), "", lf.name))
            per_role[lm.group(1)].append(t)

first = {r: min(x for x in ts if x) for r, ts in per_role.items() if any(ts)}

# ---- 2. every episodic receipt on disk ----
recs = []
for rdir in sorted((REPO/"data/agents").glob("*/episodic")):
    role = rdir.parent.name
    for f in sorted(rdir.glob("*.md")):
        dm = re.match(r"(\d{4}-\d{2}-\d{2})", f.name)
        d = dt.datetime.strptime(dm.group(1), "%Y-%m-%d") if dm else None
        recs.append({"role": role, "file": f.name, "date": d,
                     "path": str(f.relative_to(REPO))})

# ---- 3. the test ----
rows = []
for r in recs:
    role, d = r["role"], r["date"]
    fd = first.get(role)
    if fd is None:
        v, why = "NO_ROLE_DISPATCH", f"subagent_type={role} never appears in any of {len(logs)} runner logs"
    elif d is not None and d.date() < fd.date():
        v, why = "PRECEDES_FIRST_DISPATCH", f"receipt dated {d:%Y-%m-%d}; first {role} dispatch {fd:%Y-%m-%d %H:%M}"
    elif d is not None and d.date() == fd.date() and role not in {x[1] for x in dispatches if x[0] and x[0].date()==d.date()}:
        v, why = "PRECEDES_FIRST_DISPATCH", f"receipt dated {d:%Y-%m-%d}; no {role} dispatch that day"
    else:
        v, why = "backed", f"first {role} dispatch {fd:%Y-%m-%d %H:%M}"
    rows.append({**r, "date": d.strftime("%Y-%m-%d") if d else "", "verdict": v, "why": why})

OUT.mkdir(exist_ok=True)
with open(OUT/"role_identity_audit.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["role","file","path","date","verdict","why"]); w.writeheader(); w.writerows(rows)
with open(OUT/"generic_substitutions.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["when","log","subagent_type","named_roles","desc"]); w.writeheader()
    for g in sorted(generic_sub,key=lambda x:(x["when"] or dt.datetime.min)):
        w.writerow({**g,"when":g["when"].isoformat(sep=" ") if g["when"] else ""})

print(f"{len(logs)} runner logs -> {len(dispatches)} dispatch events, {len(per_role)} distinct subagent_types")
print(f"{len(recs)} episodic receipts on disk across {len({r['role'] for r in recs})} roles")
print("\nverdicts:", dict(Counter(r["verdict"] for r in rows)))
for v in ("NO_ROLE_DISPATCH","PRECEDES_FIRST_DISPATCH"):
    bad=[r for r in rows if r["verdict"]==v]
    if not bad: continue
    print(f"\n{v}  ({len(bad)}):")
    for r in bad[:40]:
        print(f"  {r['date'] or '????-??-??'}  {r['role']:24} {r['file'][:56]}")
        print(f"        {r['why']}")
print(f"\nGENERIC_SUBSTITUTION: {len(generic_sub)} dispatches with a generic subagent_type whose description names a registered role")
for g in sorted(generic_sub,key=lambda x:(x["when"] or dt.datetime.min))[:12]:
    print(f"  {g['when']:%Y-%m-%d %H:%M}  {g['subagent_type']:16} names={g['named_roles']:28} {g['desc'][:60]}")
print("\nfirst real dispatch per role:")
for r,t in sorted(first.items(),key=lambda kv:kv[1]):
    print(f"  {t:%Y-%m-%d %H:%M}  {r:24} ({len(per_role[r])} total)")
