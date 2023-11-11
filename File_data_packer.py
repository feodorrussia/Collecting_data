import os
from multiprocessing import Pool
import pandas as pd
from functools import partial


def get_versions(name, version_signal):
    with open(name + f"_v{version_signal}.txt") as file:
        text = file.read().strip().split("\n")
        add_data = text[:3]
        add_data = [list(map(float, x.split()[1:])) for x in add_data]
        # print(add_data)
        curr_data = text[3:]

    return [add_data[0][0], add_data[1][0], add_data[2][0], add_data[1][1], add_data[2][1], ' '.join(curr_data)]


def write_file_to_DB(frequency, version, num):
    path_db = "D:/Edu/Lab/Datas/filaments/maps/"
    name_database = "total_database_v2.csv"
    name_file = (f"D:/Edu/Lab/Datas/filaments/maps/{frequency}Hz/lines_v{version}/filament_map_2_{str(frequency).rjust(3, '0')}" +
                 f"Hz_{str(num).rjust(4, '0')}")  #
    if os.path.isfile(name_file + "_v0.txt"):
        input_args = [(name_file, x) for x in range(100)]
        with Pool(48) as p:
            data_add = list(p.starmap(get_versions, input_args))

        # print(data_add)
        dataframe_add = pd.DataFrame(data_add, columns=['Frequency', 'X_start', 'Y_start', 'X_end', 'Y_end', 'Values'])

        dataframe_add.to_csv(path_db + name_database, mode='a', sep=',', header=False)


if __name__ == '__main__':
    path_db = "D:/Edu/Lab/Datas/filaments/maps/"
    name_database = "total_database_v2.csv"
    data = []  # [[0.0, 0.0, 0.0, 0.0, 0.0, ("0.0; "*100)[:-2]]]
    df = pd.DataFrame(data, columns=['Frequency', 'X_start', 'Y_start', 'X_end', 'Y_end', 'Values'])
    df.to_csv(path_db + name_database, sep=',')

    # write_file_to_DB(16, 0, 12)
    #
    # df = pd.read_csv(path_db + name_database)
    # print(df.head()[['Frequency', 'X_start', 'Y_start', 'X_end', 'Y_end']])
    # print(df.head()[['Values']])

    # with Pool(1) as p:  # m-pros don't work with writing to file((
    for f in range(16, 66):
        for v in range(11):
            for n in range(28):
                write_file_to_DB(f, v, n)

    df = pd.read_csv(path_db + name_database)
    print(df.head()[['Frequency', 'X_start', 'Y_start', 'X_end', 'Y_end']])
    print(df.head()[['Values']])
