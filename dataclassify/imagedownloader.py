import urllib.request
import io
import os
import time
import sys
import errno
import calendar
import argparse
import requests
from PIL import Image

def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s    %s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

def writeto(rname, rsummary, rtime, rtotal, rfail):
    print('Generating report...')

    label = "<------------------- Report ---------------->\n\n"
    dir = "# Directory name --> " + rname + "\n\n"
    total = "# Total No.of Url's --> " + str(rtotal) + "\n\n"
    failed = "# Failed --> " + str(rfail) + "\n\n"
    time = "# Total time --> " + str(rtime) + "s"

    final_report = label + dir + total + rsummary + failed + time

    fname = rname + "_report.txt"

    try:
            file = open(fname, 'a')
            file.write(final_report)
            file.close()

    except:
        print('Something went wrong! Can\'t tell what?')
        sys.exit(0)  # quit Python
    print('Done')
    input("\nPress Enter")

#python3 imagedoenloder.py --urls=urls_2.txt --dir=class02
def main():
    parser = argparse.ArgumentParser(description='image downloder for training ')
    parser.add_argument('--urls', type=str, help='image url list textfile path')
    parser.add_argument('--dir', type=str, help='save path')

    args = parser.parse_args()
    dirname = args.dir
    urlfile = args.urls

    with open(urlfile) as f:
        print("\nDownloading...\n")

        try:
            os.makedirs(args.dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        content = f.readlines()
        i = 0

        #start = time.time()
        content = [x.strip() for x in content]
        total = len(content)
        for line in content:
            timestamp = calendar.timegm(time.gmtime())
            try:
                #urllib.request.urlretrieve(line, os.path.join(dirname, str(timestamp) + str(i) + ".jpg"))
                r = requests.get(line, timeout=4.0)
                if r.status_code != requests.codes.ok:
                    assert False, 'Status code error: {}.'.format(r.status_code)
                    print("Faile saved for {0}".format(line))
                    continue

                savepath = os.path.join(dirname, str(timestamp) + str(i) + ".jpg")
                with Image.open(io.BytesIO(r.content)) as im:
                    im.save(savepath)
                print("Image saved for {0}".format(line))
                i += 1
                progress(i, total, status='downloding')
            except:
                pass



        print("\nDownload Completed")
        #end = time.time()
        #time = end - start
        #summary = "# Total images downloaded --> " + str(i) + "\n\n"  # Failed 5
        #fail = (total) - i

    #writeto(item_name, summary, time, total, fail)

if __name__== "__main__":
    main()