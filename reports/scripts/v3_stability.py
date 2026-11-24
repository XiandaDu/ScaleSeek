"""ScaleSeek v3 (Prompt): two runs on the 7 Wiki test subsets vs the paper's GrepSeek and our reproduction."""
import json, sys
from collections import Counter
sys.path.insert(0, '/home/a32du/ScaleSeek')
from scaleseek.metrics import exact_match, f1_score, final_answer, mcnemar_exact
R = '/scratch/a32du/scaleseek_v2/runs/'
DS = ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle']
PAPER = dict(zip(DS, [40.47, 69.25, 43.63, 49.49, 46.27, 21.43, 51.20]))
PAPER_F1 = dict(zip(DS, [52.23, 76.73, 48.61, 62.31, 51.78, 30.06, 62.12]))
W = dict(zip(DS, [3610, 11313, 14267, 7405, 12576, 2417, 125]))


def load(ds, run):
    return {r['id']: r for r in map(json.loads, open(R + ds + '/' + run + '/trajectories.jsonl'))}


rows, pool = {}, {k: {} for k in ('r1', 'r2', 'gs')}
print(f"{'':16s} {'run1 EM':>8s} {'run2 EM':>8s} {'diff':>6s} {'won/lost':>9s} {'paper':>6s} {'GrepSeek':>9s} {'run1 F1':>8s} {'run2 F1':>8s} {'paper F1':>8s}")
for ds in DS:
    a, b, g = load(ds, 'scaleseek_v3_prompt__e5'), load(ds, 'scaleseek_v3_prompt_r2__e5'), load(ds, 'grepseek')
    ea = {i: exact_match(final_answer(r), r['golden_answers']) for i, r in a.items()}
    eb = {i: exact_match(final_answer(r), r['golden_answers']) for i, r in b.items()}
    eg = {i: exact_match(final_answer(r), r['golden_answers']) for i, r in g.items()}
    mc = mcnemar_exact(eb, ea)
    n = len(a)
    fa = 100 * sum(f1_score(final_answer(r), r['golden_answers']) for r in a.values()) / n
    fb = 100 * sum(f1_score(final_answer(r), r['golden_answers']) for r in b.values()) / n
    rows[ds] = (100 * sum(ea.values()) / n, 100 * sum(eb.values()) / n, fa, fb, 100 * sum(eg.values()) / n)
    print(f"{ds:16s} {rows[ds][0]:8.1f} {rows[ds][1]:8.1f} {rows[ds][1] - rows[ds][0]:+6.1f} {mc['only_a']:4d}/{mc['only_b']:<4d} {PAPER[ds]:6.2f} {rows[ds][4]:9.1f} {fa:8.1f} {fb:8.1f} {PAPER_F1[ds]:8.2f}")
    for k, e in (('r1', ea), ('r2', eb), ('gs', eg)):
        pool[k].update({ds + '/' + i: v for i, v in e.items()})
for k, j in (('run 1', 0), ('run 2', 1)):
    print(f"{k}: macro-7 EM {sum(rows[d][j] for d in DS) / 7:.2f} (paper 45.96) | weighted EM {sum(rows[d][j] * W[d] for d in DS) / sum(W.values()):.2f} (paper 49.48)"
          f" | macro-7 F1 {sum(rows[d][j + 2] for d in DS) / 7:.2f} (paper 54.83) | weighted F1 {sum(rows[d][j + 2] * W[d] for d in DS) / sum(W.values()):.2f} (paper 56.91)"
          f" | datasets above the paper: EM {sum(rows[d][j] > PAPER[d] for d in DS)}/7, F1 {sum(rows[d][j + 2] > PAPER_F1[d] for d in DS)}/7")
n = len(pool['r1'])
print(f"subset micro EM: run1 {100 * sum(pool['r1'].values()) / n:.1f}, run2 {100 * sum(pool['r2'].values()) / n:.1f}, GrepSeek {100 * sum(pool['gs'].values()) / n:.1f}")
print('run2 vs run1:', mcnemar_exact(pool['r2'], pool['r1']))
print('run2 vs GrepSeek:', mcnemar_exact(pool['r2'], pool['gs']))
print('mean of the two runs: macro-7 EM %.2f, F1 %.2f' % (sum((rows[d][0] + rows[d][1]) / 2 for d in DS) / 7, sum((rows[d][2] + rows[d][3]) / 2 for d in DS) / 7),
      '| EM per dataset', {d: round((rows[d][0] + rows[d][1]) / 2, 1) for d in DS})
