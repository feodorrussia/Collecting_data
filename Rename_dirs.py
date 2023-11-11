import os

for f in range(61, 66):
    for ver in range(11):
        if os.path.isdir(f"Datas/filaments/maps/{f}Hz/{f}Hz_v{ver}"):
            os.rename(f"Datas/filaments/maps/{f}Hz/{f}Hz_v{ver}", f"Datas/filaments/maps/{f}Hz/lines_v{ver}")