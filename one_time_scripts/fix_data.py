from pathlib import Path
import sys


def fix_line(line: str):
    line = line.replace("  ", "*&^")
    line = line.replace(" ", "")
    line = line.replace("*&^", " ")
    return line.lstrip()


def fix_file(f: Path):

    with open(f, "r") as handler:
        lines = handler.readlines()

    lines = map(fix_line, lines)
    
    with open(f, "w") as handler:
        handler.writelines(lines)

def fix_folder(p: Path):
    files = p.glob("*.txt")
    for i, f in enumerate(files):
        fix_file(f)
        if i % 100 == 0:
            print(i)

fix_folder(Path("/homes/etayl/code/bert/academic/"))

if __name__ == "__main__":
    category = sys.argv[1]
    fix_folder(Path(f"/homes/etayl/code/bert/{category}/"))