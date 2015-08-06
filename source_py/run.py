from source_py.data import DataLoader
from source_py.model import ContentFilter


def run(output_filename):
    load = DataLoader()
    model = ContentFilter(load.user_list, load.coupons_train, load.coupons_test, load.details_train)
    model.run()
    final_df = model.predict()
    final_df.to_csv(output_filename, sep=",", index=False, header=True)

if __name__ == '__main__':
    run('submissions/testing.csv')


