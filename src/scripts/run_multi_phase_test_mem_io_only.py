#!/usr/bin/env python3

import json
import subprocess
import sys
import os


def run_prmon_with_burner(burner_cmd, phase_name, expected_duration):
    print(f"Running phase: {phase_name} (expected duration: {expected_duration}s)")

    prmon_cmd = [
        "./prmon",
        "--filename",
        f"prmon_{phase_name}.txt",
        "--json-summary",
        f"prmon_{phase_name}.json",
        "--interval",
        "0.05",
        "--",
    ] + burner_cmd

    try:
        result = subprocess.run(prmon_cmd, check=True, capture_output=True, text=True)
        print(f"Phase {phase_name} completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error in phase {phase_name}: {e}")
        print(f"stderr: {e.stderr}")
        sys.exit(1)

    try:
        with open(f"prmon_{phase_name}.json") as infile:
            json_data = json.load(infile)
    except FileNotFoundError:
        print(f"prmon_{phase_name}.json not found")
        return None, None

    try:
        with open(f"prmon_{phase_name}.txt") as infile:
            lines = infile.readlines()
            headers = lines[0].strip().split("\t")
            timeseries = []
            for line in lines[1:]:
                values = line.strip().split("\t")
                row = {headers[i]: values[i] for i in range(len(values))}
                timeseries.append(row)
    except FileNotFoundError:
        print(f"prmon_{phase_name}.txt not found")
        timeseries = None

    return json_data, timeseries


def main():
    print(
        "Starting multi-phase anomaly detection test (mem-burner and io-burner only)..."
    )

    os.chdir("/home/atharva20240519/prmon/build/package")

    phase1_cmd = [
        "./tests/io-burner",
        "--io",
        "1",
        "--threads",
        "1",
        "--usleep",
        "10000",
        "--pause",
        "300",
    ]
    phase1_json, phase1_ts = run_prmon_with_burner(phase1_cmd, "baseline1", 300)

    phase2_cmd = ["./tests/mem-burner", "--malloc", "4096", "--sleep", "180"]
    phase2_json, phase2_ts = run_prmon_with_burner(phase2_cmd, "memory_spike", 180)

    phase3_cmd = [
        "./tests/io-burner",
        "--io",
        "1",
        "--threads",
        "1",
        "--usleep",
        "10000",
        "--pause",
        "300",
    ]
    phase3_json, phase3_ts = run_prmon_with_burner(phase3_cmd, "recovery1", 300)

    phase4_cmd = [
        "./tests/io-burner",
        "--io",
        "10",
        "--threads",
        "16",
        "--usleep",
        "1",
        "--pause",
        "180",
    ]
    phase4_json, phase4_ts = run_prmon_with_burner(phase4_cmd, "cpu_surge", 180)

    phase5_cmd = [
        "./tests/io-burner",
        "--io",
        "1",
        "--threads",
        "1",
        "--usleep",
        "10000",
        "--pause",
        "300",
    ]
    phase5_json, phase5_ts = run_prmon_with_burner(phase5_cmd, "recovery2", 300)

    phase6_cmd = [
        "./tests/io-burner",
        "--io",
        "400",
        "--threads",
        "4",
        "--usleep",
        "10",
        "--pause",
        "180",
    ]
    phase6_json, phase6_ts = run_prmon_with_burner(phase6_cmd, "io_burst", 180)

    phase7_cmd = [
        "./tests/io-burner",
        "--io",
        "1",
        "--threads",
        "1",
        "--usleep",
        "10000",
        "--pause",
        "360",
    ]
    phase7_json, phase7_ts = run_prmon_with_burner(phase7_cmd, "final_baseline", 360)

    combined_dataset = {
        "phases": {
            "baseline1": phase1_json,
            "memory_spike": phase2_json,
            "recovery1": phase3_json,
            "cpu_surge": phase4_json,
            "recovery2": phase5_json,
            "io_burst": phase6_json,
            "final_baseline": phase7_json,
        }
    }

    combined_filename = "prmon_combined_dataset.json"
    with open(combined_filename, "w") as outfile:
        json.dump(combined_dataset, outfile, indent=2)

    print(f"\nCombined dataset saved to: {combined_filename}")

    combined_timeseries = {
        "phases": {
            "baseline1": phase1_ts,
            "memory_spike": phase2_ts,
            "recovery1": phase3_ts,
            "cpu_surge": phase4_ts,
            "recovery2": phase5_ts,
            "io_burst": phase6_ts,
            "final_baseline": phase7_ts,
        }
    }

    timeseries_filename = "prmon_combined_timeseries.json"
    with open(timeseries_filename, "w") as outfile:
        json.dump(combined_timeseries, outfile, indent=2)

    print(f"Combined time-series saved to: {timeseries_filename}")


if __name__ == "__main__":
    main()
