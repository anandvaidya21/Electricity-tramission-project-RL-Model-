#!/usr/bin/env python3
"""Pre-submission validation script"""

import sys
import os
import subprocess
import json
from pathlib import Path

def check_file_exists(path: str, name: str) -> bool:
    """Check if required file exists"""
    if Path(path).exists():
        print(f"✓ {name}")
        return True
    print(f"✗ {name} - NOT FOUND")
    return False

def check_environment_variables() -> bool:
    """Check required env vars"""
    vars = ["API_BASE_URL", "MODEL_NAME", "HF_TOKEN"]
    all_set = True
    for var in vars:
        if os.getenv(var):
            print(f"✓ {var} set")
        else:
            print(f"✗ {var} NOT SET")
            all_set = False
    return all_set

def check_docker_build() -> bool:
    """Check Dockerfile builds"""
    try:
        result = subprocess.run(
            ["docker", "build", "-t", "electricity-env:test", "."],
            capture_output=True,
            timeout=300
        )
        if result.returncode == 0:
            print("✓ Docker builds successfully")
            return True
        print("✗ Docker build failed")
        return False
    except Exception as e:
        print(f"✗ Docker check error: {e}")
        return False

def check_inference_runs() -> bool:
    """Check inference.py runs without error"""
    try:
        result = subprocess.run(
            ["python", "inference.py"],
            capture_output=True,
            timeout=1200  # 20 minutes
        )
        if result.returncode == 0:
            print("✓ inference.py runs successfully")
            return True
        print(f"✗ inference.py failed: {result.stderr.decode()}")
        return False
    except subprocess.TimeoutExpired:
        print("✗ inference.py exceeded 20 minute timeout")
        return False
    except Exception as e:
        print(f"✗ inference.py error: {e}")
        return False

def check_openenv_yaml() -> bool:
    """Validate openenv.yaml"""
    try:
        import yaml
        with open("openenv.yaml") as f:
            spec = yaml.safe_load(f)
        
        required_keys = ["name", "version", "environment", "observation", "action", "tasks"]
        if all(key in spec for key in required_keys):
            print("✓ openenv.yaml is valid")
            return True
        print(f"✗ openenv.yaml missing keys: {set(required_keys) - set(spec.keys())}")
        return False
    except Exception as e:
        print(f"✗ openenv.yaml error: {e}")
        return False

def main():
    """Run all validations"""
    print("\n========== PRE-SUBMISSION VALIDATION ==========\n")
    
    checks = [
        ("File Structure", lambda: all([
            check_file_exists("inference.py", "inference.py in root"),
            check_file_exists("openenv.yaml", "openenv.yaml"),
            check_file_exists("Dockerfile", "Dockerfile"),
            check_file_exists("README.md", "README.md"),
            check_file_exists("requirements.txt", "requirements.txt"),
        ])),
        ("Environment Variables", check_environment_variables),
        ("OpenEnv Spec", check_openenv_yaml),
        ("Docker Build", check_docker_build),
        ("Inference Script", check_inference_runs),
    ]
    
    results = {}
    for name, check in checks:
        print(f"\n--- {name} ---")
        try:
            results[name] = check()
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            results[name] = False
    
    print("\n========== SUMMARY ==========")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"{passed}/{total} checks passed\n")
    
    if passed == total:
        print("✓ READY FOR SUBMISSION")
        return 0
    else:
        print("✗ FIX FAILURES BEFORE SUBMISSION")
        return 1

if __name__ == "__main__":
    sys.exit(main())