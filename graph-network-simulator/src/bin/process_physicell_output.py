import argparse

from physicell import get_physicell_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--physicell-output", required=True)
    parser.add_argument("-o", "--output-file", required=True)
    args = parser.parse_args()
    physicell_out_df = get_physicell_df(args.physicell_output)
    physicell_out_df.to_csv(args.output_file, index=False)


if __name__ == '__main__':
    main()
