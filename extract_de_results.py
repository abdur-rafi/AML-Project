import re
import csv
import os

def extract_de_results(log_path, output_csv=None):
    de_acc_re = re.compile(r"DE:\d+")
    metrics_re = re.compile(r"metrics: OrderedDict\(\[.*?'top1', ([\d\.]+).*?'f1', ([\d\.]+).*?\]\)")
    metrics_f1_re = re.compile(r"metrics_f1:\s*([\d\.]+)")
    de_acc_line_re = re.compile(r"DE:\d+ Acc@1: ([\d\.]+)")

    # Read all lines
    with open(log_path, "r") as f:
        lines = f.readlines()

    # Find indices of DE lines
    de_indices = [i for i, line in enumerate(lines) if de_acc_re.search(line)]
    # Add start and end boundaries
    boundaries = [0] + de_indices + [len(lines)]

    max_top1_overall = None
    max_f1_overall = None
    csv_rows = []

    # For each section between DE lines
    for b in range(len(boundaries)-1):
        start = boundaries[b]
        end = boundaries[b+1]
        top1s = []
        f1s = []
        de_acc = None
        for i in range(start, end):
            m = metrics_re.search(lines[i])
            if m:
                top1 = float(m.group(1))
                f1 = float(m.group(2))
                top1s.append(top1)
                f1s.append(f1)
            m_f1 = metrics_f1_re.search(lines[i])
            if m_f1:
                f1_val = float(m_f1.group(1))
                f1s.append(f1_val)
            m_de_acc = de_acc_line_re.search(lines[i])
            if m_de_acc:
                de_acc = float(m_de_acc.group(1))
        # Find max in this section
        max_top1 = max(top1s) if top1s else None
        max_f1 = max(f1s) if f1s else None
        # If DE acc is present and higher, use it
        if de_acc is not None and (max_top1 is None or de_acc > max_top1):
            max_top1 = de_acc
        if max_top1 is not None and (max_top1_overall is None or max_top1 > max_top1_overall):
            max_top1_overall = max_top1
        if max_f1 is not None and (max_f1_overall is None or max_f1 > max_f1_overall):
            max_f1_overall = max_f1
        csv_rows.append([max_top1, max_f1])

    # Write to CSV
    if output_csv is None:
        csv_path = os.path.join(os.path.dirname(log_path), "de_results.csv")
    else:
        csv_path = output_csv
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Accuracy", "F1"])
        for row in csv_rows:
            writer.writerow(row)
    print(f"CSV written to {csv_path}")
    print(f"Overall Maximum top1: {max_top1_overall}")
    print(f"Overall Maximum f1: {max_f1_overall}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) not in [2, 3]:
        print("Usage: python extract_de_results.py <log_file> [output_csv]")
        sys.exit(1)
    log_file = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) == 3 else None
    extract_de_results(log_file, output_csv)