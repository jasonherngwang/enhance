# Usage: seam_check.py <video> [join_frame ...]
# join_frame = a stitched-output frame index where two chunks meet.
# A join is flagged SEAM if its inter-frame pixel diff exceeds 2x the clip's median.
import sys, cv2, numpy as np, statistics
f = sys.argv[1]
joins = [int(x) for x in sys.argv[2:]] or [160, 320]
c = cv2.VideoCapture(f); fr = []
while True:
    ok, x = c.read()
    if not ok: break
    fr.append(cv2.resize(x, (416, 240)))
diffs = [float(np.mean(cv2.absdiff(fr[i], fr[i+1]))) for i in range(len(fr)-1)]
med = statistics.median(diffs)
print(f"{f}: frames={len(fr)} median inter-frame diff={med:.3f}")
for b in joins:
    if 0 < b < len(fr):
        r = diffs[b-1] / med
        tag = "SEAM" if r > 2 else "clean"
        print(f"  join@{b}: {diffs[b-1]:.3f} ({r:.2f}x median) -> {tag}")
# also worst-5 frames anywhere, to see if joins even rank
worst = sorted(range(len(diffs)), key=lambda i: -diffs[i])[:5]
print("  worst-5 diff positions:", [(i+1, round(diffs[i],2)) for i in worst])
