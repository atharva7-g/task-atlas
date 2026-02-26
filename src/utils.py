from pathlib import Path

def get_project_root() -> Path:
	parent = Path(__file__).resolve().parent.parent
	return parent

if __name__ == "__main__":
	parent = get_project_root()
	print(parent)

