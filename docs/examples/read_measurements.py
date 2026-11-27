#!/usr/bin/env python3
"""Read a measurement file (schema version 2) and answer: where did the time go?

Standard library only. Usage:
    python3 read_measurements.py softmax_matmul.json [how_many_nodes]

Prints the job's headline numbers, the most expensive nodes by GPU compute,
and how much of each node's wall time was spent waiting rather than computing.
"""

import json
import sys

SUPPORTED_SCHEMA_VERSION = 2
WAITING_SLICES = ("wait_on_host_us", "wait_on_own_stream_us", "wait_on_other_stream_us",
                  "gpu_scheduling_us")


def load_measurements(measurement_path):
    with open(measurement_path, encoding="utf-8") as measurement_file:
        measurements = json.load(measurement_file)
    if measurements.get("kind") == "comparison":
        sys.exit("this is a comparison record; read one of the variant files it names "
                 "(baseline.measurement_file, candidate.measurement_file)")
    if measurements.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        sys.exit(f"schema_version {measurements.get('schema_version')} is not "
                 f"{SUPPORTED_SCHEMA_VERSION}; fields may have changed meaning")
    return measurements


def node_cost_microseconds(node_record):
    """A node's own GPU compute, or None when it was not measured on its own.

    In a per_stage run nodes are timed as part of their stage, so compute_us
    is null and the stage's record holds the time; serial_duration_us is the
    node run alone. null always means "not measured", never zero.
    """
    return node_record["compute_us"]


def print_job_headline(measurements):
    job = measurements["job"]
    print(f"{measurements['variant_name']}: {job['executed_node_count']} nodes on "
          f"{job['stream_count']} streams, {measurements['granularity']} timing")
    print(f"  job duration {job['job_wall_us']['median']:.1f} us median "
          f"({job['job_wall_us']['min']:.1f} to {job['job_wall_us']['max']:.1f})")
    print(f"  GPU execution {job['measured_gpu_execution_us']:.1f} us, "
          f"cross-stream stall {job['cross_stream_stall_us']:.1f} us")


def print_most_expensive_nodes(measurements, how_many_nodes):
    timed_nodes = [node for node in measurements["nodes"] if node_cost_microseconds(node) is not None]
    if not timed_nodes:
        print("  no node was timed on its own (a per_stage run): see measurements['stages']")
        return
    timed_nodes.sort(key=node_cost_microseconds, reverse=True)
    print(f"  most expensive {min(how_many_nodes, len(timed_nodes))} nodes by GPU compute:")
    for node in timed_nodes[:how_many_nodes]:
        waiting_microseconds = sum(node[slice_name] or 0.0 for slice_name in WAITING_SLICES)
        kernel = f" ({node['kernel_name']})" if node["kernel_name"] else ""
        print(f"    {node['label']:>4} {node['op_name']}{kernel}: compute {node['compute_us']:.1f} us, "
              f"waiting {waiting_microseconds:.1f} us, mostly {node['classification']}")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    measurements = load_measurements(sys.argv[1])
    how_many_nodes = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    print_job_headline(measurements)
    print_most_expensive_nodes(measurements, how_many_nodes)


if __name__ == "__main__":
    main()
