#!/usr/bin/env python

#
# Writes emails into a csv table with the following fields:
#
#   path, date, message-id, from, to, to-type
#
# Where 'to-type' is one of 'To', 'Cc', 'Bcc', 'X-To'
# Currently, 'X-To' is only written to the table for emails that have no 'To',
# 'Cc' or 'Bcc' field filled in, and that have an 'X-To' to "All Enron
# Worldwide". 'X-cc' and 'X-bcc' are completely ignored, at the moment, as is
# 'X-To' if any of the regular "to" fields were filled.
#

import os
import re
import argparse
import email.parser
import csv

def process_mails(outfile, rootdir):

    count = 0 # for progress reporting...
    skipped_no_to = 0
    skipped_x_to = 0

    parser = email.parser.Parser()

    with open(outfile, "wb") as outf:
        csvwriter = csv.writer(outf)
        for path, dirs, files in os.walk(rootdir):
            for fname in files:
                with open(os.path.join(path, fname)) as f:

                    hdrs = parser.parse(f, True)

                mailpath = os.path.join(path[len(rootdir):], fname)

                outelems = []
                outelems.append(mailpath)
                outelems.append(hdrs['Date'])
                outelems.append(hdrs['Message-ID'])
                outelems.append(hdrs['From'])
                # for each of the "to" fields, add a separate row for each
                # recipient, and the "to-type"
                written = False
                for to in ['To', 'Cc', 'Bcc']:
                    if hdrs[to]:
                        for rcpt in hdrs[to].split(','):
                            csvwriter.writerow(outelems + [rcpt.strip(), to])
                            written = True    
                if not written:
                    # is this a company-wide email?
                    if hdrs['X-to'] and "All Enron Worldwide" in hdrs['X-To']:
                        csvwriter.writerow(outelems + ['All Enron Worldwide', 'X-To'])
                    else:
                        if hdrs['X-To'] or hdrs['X-cc'] or hdrs['X-bcc']:
                            xtos = hdrs['X-To'] + hdrs['X-cc'] + hdrs['X-bcc']
                            print "Skipping %s, has X-to fields: %s" % (mailpath, xtos)
                            skipped_x_to += 1
                        else:
                            skipped_no_to += 1

                # progress reporting
                count += 1
                if count % 100 == 0:
                    print count, "(skipped: %d,%d)" % (skipped_x_to, skipped_no_to)

def main():

    # handle args

    parser = argparse.ArgumentParser(description='Parse Enron emails into structured csv')
    parser.add_argument('outfile', type=str, help='name of output file')
    parser.add_argument('--rootdir', type=str, default='.', help='path to the maildir')

    args = parser.parse_args()

    process_mails(args.outfile, args.rootdir)
    
if __name__ == '__main__':
    main()

