import pandas as pd

user_list = pd.read_csv("raw_data/user_list.csv", header=0)


if __name__ == '__main__':

    print user_list.shape

    print user_list.head()
