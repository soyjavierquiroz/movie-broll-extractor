import json
from pathlib import Path
from movie_broll.cli import main
def test_cli_help_and_missing_paths(capsys, tmp_path):
 try: main(["--help"])
 except SystemExit as exc: assert exc.code==0
 assert main(["inspect","--movie",str(tmp_path/'missing.mp4'),"--srt",str(tmp_path/'missing.srt'),"--run-dir",str(tmp_path/'run')])==2
 (tmp_path/"movie.mp4").write_bytes(b"x")
 assert main(["inspect","--movie",str(tmp_path/'movie.mp4'),"--srt",str(tmp_path/'missing.srt'),"--run-dir",str(tmp_path/'run2')])==2
def test_schema_json():
 root=Path(__file__).parents[1]/"schemas"
 for path in root.glob("*.json"): assert json.loads(path.read_text())["$schema"]
