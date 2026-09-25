import os, glob
print("=== /kaggle/input tree ===")
for root, dirs, files in os.walk("/kaggle/input"):
    depth = root.count(os.sep) - 2
    if depth <= 2:
        print("  "*depth + root + "/")
        for f in files[:6]:
            print("  "*(depth+1) + f)
print("=== all tars ===")
print(glob.glob("/kaggle/input/**/*.tar", recursive=True))
print("=== all train.csv ===")
print(glob.glob("/kaggle/input/**/train.csv", recursive=True)[:5])
