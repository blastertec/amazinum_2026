import argparse
import csv
import json

from forecasting import forecast, AIRLINE


def load_csv(path, column=None):
    with open(path) as f:
        rows = [r for r in csv.reader(f) if r]
    header = rows[0]
    try:
        float(header[-1])          # no header row
        start, idx = 0, len(header) - 1
    except ValueError:             # first row is a header
        start = 1
        idx = header.index(column) if column in header else len(header) - 1
    return [float(r[idx]) for r in rows[start:]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input")
    p.add_argument("--column")
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--seasonal-periods", type=int)
    p.add_argument("--method", default="auto")
    p.add_argument("--sample", action="store_true")
    args = p.parse_args()

    if args.input and not args.sample:
        values = load_csv(args.input, args.column)
        seasonal = args.seasonal_periods
    else:
        values = AIRLINE
        seasonal = args.seasonal_periods or 12

    preds, method = forecast(values, args.steps, seasonal, args.method)
    print(json.dumps({"method_used": method, "steps": args.steps, "forecast": preds}, indent=2))


if __name__ == "__main__":
    main()
