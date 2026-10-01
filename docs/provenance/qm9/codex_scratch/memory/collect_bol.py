#!/usr/bin/env python3
"""Collect best-of-lookahead results. Handles both schemas:
  old (unfiltered): runs final_endpoint, best_lookahead
  new (PB-valid)  : runs final_endpoint, best_lookahead_raw, best_lookahead_pbvalid
Reports oracle MAE + PB for final vs selected, guide MAE (Goodhart gap), vs baseline."""
import json, glob, os, sys
H=27211.386245988; EN={"gap","homo","lumo"}
BASE={"cv":0.739,"mu":0.959,"alpha":1.983,"gap":678.0,"homo":284.0,"lumo":450.0}
PB_FLOOR=0.90
root=sys.argv[1] if len(sys.argv)>1 else os.path.dirname(__file__)
rows=[]
for p in glob.glob(os.path.join(root,"**","summary.json"),recursive=True):
    try: d=json.load(open(p))
    except Exception: continue
    a=d.get("args",{}); prop=a.get("property_name")
    if prop not in BASE: continue
    S={r["run"]:r for r in d["summary"]}
    fin=S.get("final_endpoint")
    sel=S.get("best_lookahead_pbvalid") or S.get("best_lookahead")
    raw=S.get("best_lookahead_raw")
    if not fin or not sel: continue
    conv=lambda x: x*H if prop in EN else x
    rows.append(dict(prop=prop, mu=float(a.get("mu",0)), selmint=float(a.get("select_min_t",0)),
                     fin_o=conv(fin["oracle_mae"]), fin_pb=fin.get("pb_pb_valid"),
                     sel_g=conv(sel["guide_mae"]), sel_o=conv(sel["oracle_mae"]), sel_pb=sel.get("pb_pb_valid"),
                     raw_o=conv(raw["oracle_mae"]) if raw else None,
                     raw_pb=raw.get("pb_pb_valid") if raw else None,
                     fracvalid=sel.get("frac_with_valid_candidate"),
                     selt=sel.get("selected_t_mean"),
                     base=BASE[prop], tag=os.path.basename(os.path.dirname(p))))
order={p:i for i,p in enumerate(["cv","mu","alpha","gap","homo","lumo"])}
rows.sort(key=lambda r:(order.get(r["prop"],9), r["selmint"]))
u=lambda p:"meV" if p in EN else ""
print(f"{'prop':5} {'mu':>5} {'selT':>4} | {'FINoMAE':>8} {'FINpb':>5} | {'RAWoMAE':>8} {'RAWpb':>5} | {'PBVoMAE':>8} {'PBVpb':>5} {'selt̄':>4} | {'base':>7} {'impr%':>6} {'':>5}")
print("-"*118)
for r in rows:
    imp=(r["base"]-r["sel_o"])/r["base"]*100
    beat="<BEAT" if (r["sel_o"]<r["base"] and r["sel_pb"] is not None and r["sel_pb"]>=PB_FLOOR) else ("lowPB" if (r["sel_pb"] is not None and r["sel_pb"]<PB_FLOOR) else "")
    f=lambda x,unit=True: (f"{x:7.2f}"+(u(r['prop']) if unit else "")) if x is not None else "   -   "
    g=lambda x: f"{x:.3f}" if x is not None else "  -  "
    st=f"{r['selt']:.2f}" if r['selt'] is not None else "  - "
    print(f"{r['prop']:5} {r['mu']:5.2f} {r['selmint']:4.2f} | {f(r['fin_o'])} {g(r['fin_pb']):>5} | {f(r['raw_o'])} {g(r['raw_pb']):>5} | {f(r['sel_o'])} {g(r['sel_pb']):>5} {st:>4} | {r['base']:5.1f}{u(r['prop']):>2} {imp:5.1f}% {beat:>5}")
