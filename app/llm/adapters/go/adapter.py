import subprocess
import tempfile
import os

def run_go_code(code: str) -> dict:
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "main.go")
        with open(filepath, "w") as f:
            f.write(code)

        build = subprocess.run(["go", "build", "-o", "main", filepath], cwd=tmpdir, capture_output=True, text=True)
        if build.returncode != 0:
            return {"success": False, "stderr": build.stderr, "stdout": build.stdout, "exit_code": build.returncode}

        run = subprocess.run(["./main"], cwd=tmpdir, capture_output=True, text=True)
        return {
            "success": run.returncode == 0,
            "stdout": run.stdout,
            "stderr": run.stderr,
            "exit_code": run.returncode,
        }
