"""A zone of radius zero: a later task with the instruction of an older task, word for word (Definition 1 of the
paper: such a zone holds only inputs at distance zero, and ties go to the older task).
Run: python test_zero_radius.py   (needs only torch; no model is loaded)"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from certified_switch import Zones

torch.manual_seed(0)
V, d = 300, 16
table = torch.randn(V, d)
g = torch.Generator().manual_seed(1)
def rows(opening, n, length=12):
    return [list(opening) + torch.randint(1, V, (length,), generator=g).tolist() for _ in range(n)]
unrelated = [rows(torch.randint(1, V, (8,), generator=g).tolist(), 30) for _ in range(40)]
z = Zones(table, pad_id=0)
z.fit_background(unrelated)
op1 = torch.randint(1, V, (8,), generator=g).tolist()
op2 = torch.randint(1, V, (8,), generator=g).tolist()
t1, t2 = rows(op1, 20), rows(op2, 20)
assert z.add_task(t1) == 0
assert z.add_task(t2) == 0
t3 = rows(op1, 20)                      # a third task with the instruction of task 1, word for word
blocked = z.add_task(t3)
print("blocked inputs of the third task:", blocked, "of", len(t3), "| its radius:", z.tasks[2]["rho"])
assert z.tasks[2]["rho"] == 0.0 and blocked == len(t3)
for rule in ("certified", "nearest", "exact_first"):
    r, c = z.route(t3, rule)
    print(rule, "routes:", sorted(set(r.tolist())), "certified:", int(c.sum()), "of", len(t3))
    assert set(r.tolist()) == {1}, rule          # the older task keeps them under every rule (ties go to the older task)
r, c = z.route(t2, "certified"); assert set(r.tolist()) == {2} and bool(c.all())
r, c = z.route(rows(torch.randint(1, V, (8,), generator=g).tolist(), 5), "nearest"); assert not torch.isnan(r.float()).any()
print("ok")
