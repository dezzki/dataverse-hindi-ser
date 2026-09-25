import os, glob
print("=== recursive all files under /kaggle/input ===")
for p in glob.glob("/kaggle/input/**/*", recursive=True):
    if os.path.isfile(p):
        print("FILE", os.path.getsize(p), p)
print("=== datasets dirs ===")
for d in glob.glob("/kaggle/input/datasets/**", recursive=True):
    print("DIR", d)
print("=== competitions dirs ===")
for d in glob.glob("/kaggle/input/competitions/**", recursive=True)[:10]:
    print("DIR", d)
