import subprocess
import tempfile
import os

def run_js_code(code: str) -> dict:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
        f.write(code)
        f.flush()
        path = f.name

    try:
        result = subprocess.run(
            ["node", path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
        }
    finally:
        os.remove(path)
