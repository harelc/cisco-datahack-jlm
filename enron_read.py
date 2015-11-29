
import os
import re
import argparse
import email.parser
import random
import csv

def process_mails(rootdirs, fields = ['Date','From'], ratio=1.0):
    parser = email.parser.Parser()
    for rootdir in rootdirs:
        for path, dirs, files in os.walk(rootdir):
            print path
            for fname in files:
                if not random.random() < ratio: continue
                with open(os.path.join(path, fname)) as f:
                    hdrs = parser.parse(f, True)
                body = hdrs.get_payload()
                if '---' in body:
                    body = body[:body.index('---')]
                yield [hdrs[field] for field in fields] + [body]
