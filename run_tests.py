import subprocess
from pathlib import Path
import pandas as pd
import sys
from datetime import datetime

def ConvertHumanTimeToSeconds(time_list):
    seconds = 0.0

    for t in time_list:
        if "ns" in t:
            time_string = t.split("ns")[0]
            seconds += float(time_string) * 1e-9
        elif "micros" in t:
            time_string = t.split("micros")[0]
            seconds += float(time_string) * 1e-6
        elif "ms" in t:
            time_string = t.split("ms")[0]
            seconds += float(time_string) * 1e-3
        elif "s" in t:
            time_string = t.split("s")[0]
            seconds += float(time_string)
        elif "min" in t:
            time_string = t.split("min")[0]
            seconds += float(time_string) * 60
        elif "h" in t:
            time_string = t.split("h")[0]
            seconds += float(time_string) * 3600
        elif "d" in t:
            time_string = t.split("d")[0]
            seconds += float(time_string) * 3600 * 24

    return seconds

generate_new = False


matrix_types = ["semidefinite", "spn", "spn_2_neg_ev", "perfect_cop"]
dimensions = [3,4,5,6,7,8]

try:
    df = pd.read_csv("results", index_col = 0)
except:
    print("generating new empty df")

    data = {
        "matrix name": [],
        "matrix type": [],
        "matrix dimension": [],
        "matrix number": [],
        "already handled": [],
        "one_ev_code time": [],
        "spn_code time": [],
        "simplex partitioning code time": []
    }

    for matrix_type in matrix_types:
        if matrix_type == "spn_2_neg_ev":
            dimensions = [5,6,7,8]
        else:
            dimensions = [3,4,5,6,7,8]
        for d in dimensions:
            p = Path(f"./test_matrices/{matrix_type}/{d}")
            file_names = [x.resolve() for x in p.iterdir()]

            for file_name in file_names:
                matrix_number = str(file_name).split(f"cop{d}_")[-1]
                matrix_name = str(file_name).split("test_matrices/")[-1]
                print(f"matrix name {matrix_name}")

                already_handled = False
                time_one_ev = -2.0
                time_spn = -2.0
                time_simplex = -2.0

                data["matrix name"].append(matrix_name)
                data["matrix type"].append(matrix_type)
                data["matrix dimension"].append(d)
                data["matrix number"].append(int(matrix_number))
                data["already handled"].append(already_handled)
                data["one_ev_code time"].append(time_one_ev)
                data["spn_code time"].append(time_spn)
                data["simplex partitioning code time"].append(time_simplex)

    df = pd.DataFrame(data)
    df.to_csv("results")
    print("done. Rerun to start running.")
    sys.exit()





#-1 is error in execution
#-2 is not yet handled
#-3 is time taken too long

for matrix_type in matrix_types:
    if matrix_type == "spn_2_neg_ev":
        dimensions = [5,6,7,8]
    else:
        dimensions = [3,4,5,6,7,8]
    for d in dimensions:
        p = Path(f"./test_matrices/{matrix_type}/{d}")
        file_names = [x.resolve() for x in p.iterdir()]

        for file_name in file_names:
            matrix_number = str(file_name).split(f"cop{d}_")[-1]
            matrix_name = str(file_name).split("test_matrices/")[-1]

            row = df.loc[df['matrix name'] == matrix_name]
            i = row.index[0]

            if row['already handled'].values[0]:
                continue

            print(f"[{datetime.now()}] working on {matrix_name}.. mit index {i}")
            try:
                o = subprocess.run(args = ["./CP_CopositiveMin", "gmp", file_name], capture_output = True, text = True, timeout = 1 * 60 * 120)

                if "runtime = " in o.stderr:
                    time_list = o.stderr.rstrip().split("runtime = ")[1].split(" timeanddate")[0].split(" ")
                    time_simplex = ConvertHumanTimeToSeconds(time_list)
                else:
                    print(f"fehler in simplex bei {file_name}:")
                    print(o.stderr)
                    time_simplex = -1.0
            except subprocess.TimeoutExpired:
                time_simplex = -3.0

            o = subprocess.run(args = ["python3", "src/calculate_cop_min.py", "SPN", file_name], capture_output = True, text = True)
            if "insgesamt" in o.stdout:
                time_spn = float(((o.stdout.rstrip().split("insgesamt, "))[1].split(" in main"))[0])
            else:
                time_spn = -1.0

            o = subprocess.run(args = ["python3", "src/calculate_cop_min.py", "O", file_name], capture_output = True, text = True)
            if "insgesamt" in o.stdout:
                time_one_ev = float(((o.stdout.rstrip().split("insgesamt, "))[1].split(" in main"))[0])
            else:
                time_one_ev = -1.0


            df.at[i, 'already handled'] = True
            df.at[i, 'one_ev_code time'] = time_one_ev
            df.at[i, 'spn_code time'] = time_spn
            df.at[i, 'simplex partitioning code time'] = time_simplex

            df.to_csv("results")

        print(f"Dimension {d} bei Type {matrix_type} done")
