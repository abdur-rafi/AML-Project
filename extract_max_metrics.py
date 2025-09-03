import re
import csv
import os

def extract_metrics(log_path):
    metrics_re = re.compile(r"metrics: OrderedDict\(\[.*?'top1', ([\d\.]+).*?'f1', ([\d\.]+).*?\]\)")
    metrics_f1_re = re.compile(r"metrics_f1: ([\d\.]+)")

    top1_values = []
    f1_values = []

    with open(log_path, "r") as f:
        for line in f:
            m = metrics_re.search(line)
            if m:
                top1 = float(m.group(1))
                f1 = float(m.group(2))
                top1_values.append(top1)
                f1_values.append(f1)
            m_f1 = metrics_f1_re.search(line)
            if m_f1:
                f1_val = float(m_f1.group(1))
                # Only add f1 if not already added from metrics line
                if not (f1_values and abs(f1_values[-1] - f1_val) < 1e-4):
                    f1_values.append(f1_val)

    max_top1 = max(top1_values) if top1_values else None
    max_f1 = max(f1_values) if f1_values else None

    print(f"Maximum top1: {max_top1}")
    print(f"Maximum metrics_f1: {max_f1}")

    # Write to CSV
    csv_path = os.path.join(os.path.dirname(log_path), "metrics.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["top1", "f1"])
        # Pad shorter list with empty values
        for i in range(max(len(top1_values), len(f1_values))):
            top1 = top1_values[i] if i < len(top1_values) else ""
            f1 = f1_values[i] if i < len(f1_values) else ""
            writer.writerow([top1, f1])
    print(f"CSV written to {csv_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python extract_max_metrics.py <log_file>")
        sys.exit(1)
    log_file = sys.argv[1]
    extract_metrics(log_file)