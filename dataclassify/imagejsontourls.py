
import sys
import argparse
import json


def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s    %s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

def main():
    parser = argparse.ArgumentParser(description='image downloder for training ')
    parser.add_argument('--jsonfile', type=str, help='json file path for parsing')
    parser.add_argument('--urlsfile', type=str, help='image url text file path')

    args = parser.parse_args()

    jsonfile = args.jsonfile
    urlsfile = args.urlsfile

    i = 0

    with open(jsonfile, 'r') as f:
        data = json.load(f)

    filew = open(urlsfile, "w+")
    for p in data:
        stitle =  p['stitle']
        if stitle.find('soccer') != -1 or stitle.find('football') != -1:
            surl =  p['thumbnail']
            filew.write(surl)
            filew.write("\n")
        i = i + 1
        #progress(i, total, status='downloding')

    filew.close()

    print("\nTask Completed")


if __name__== "__main__":
    main()
