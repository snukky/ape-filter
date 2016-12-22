#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import re

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__),
                           "Evaluation_Script/tercom-0.7.25")

SPLIT_REGEX = re.compile(r"\s*\|\s*")

def main():
    args = parse_user_args()

    # Adding sentence id to the text files
    ter_ref = add_sentence_id(args.reference, id=args.identifier)
    ter_hyp = add_sentence_id(args.hypothesis, id=args.identifier)

    # Run TER
    ter, out = run_evaluation(args.hypothesis, args.reference, args.identifier,
                              not args.case_insensitive)

    if args.detailed:
        table = parse_result_table(out)
        for vals in table:
             print "\t".join(str(v) for v in vals)
    else:
        print "{}: {:.4f}".format(args.identifier, ter)

    if not args.no_clean:
        if os.path.exists(ter_ref):
            os.remove(ter_ref)
        if os.path.exists(ter_hyp):
            os.remove(ter_hyp)
        os.remove(out)

def add_sentence_id(file, id):
    output = "{}_{}_ter".format(file, id)
    if not os.path.exists(output):
        os.popen("python {scripts}/AddSentenceId.py {file} {out} {id}" \
            .format(scripts=SCRIPTS_DIR, file=file, out=output, id=id))
    return output

def run_evaluation(hyp, ref, id, case=True):
    ref_base = os.path.split(ref)[-1]
    out_file = "{}_{}_ter-{}-eval_case".format(hyp, id, ref_base)
    out_file += "Sens" if case else "Insens"

    if not os.path.exists(out_file + ".sum"):
        cmd = "java -jar {scripts}/tercom.7.25.jar {case} " \
            "-r {ref}_{id}_ter -h {hyp}_{id}_ter -n {out} -o sum" \
            .format(scripts=SCRIPTS_DIR, case="-s" if case else "",
                    hyp=hyp, ref=ref, id=id, out=out_file)
        # print >>sys.stderr, cmd
        os.popen(cmd)

    with open(out_file + ".sum") as f:
        result_line = f.readlines()[-1].strip()
    score = float(result_line.split()[-1].replace(",", "."))
    return score, out_file + ".sum"

def parse_result_table(file):
    with open(file) as f:
        output = f.readlines()
    header = SPLIT_REGEX.split(output[3].strip())[1:]
    table = [header]
    for line in output[5:]:
        if line.startswith("---"):
            continue
        fields = SPLIT_REGEX.split(line.strip().replace(",", "."))
        values = [int(float(v)) for v in fields[1:-1]] + [float(fields[-1])]
        table.append(values)
    return table

def parse_user_args():
    parser = argparse.ArgumentParser(
        description="This script computes the HTER of an input hypothesis file. " \
                    "It is based on the official script runHTER.sh")
    parser.add_argument("-s", "--hypothesis", required=True)
    parser.add_argument("-r", "--reference", required=True)

    parser.add_argument("-i", "--identifier", default="APE")
    parser.add_argument("-c", "--case-insensitive", action='store_true')
    parser.add_argument("-d", "--detailed", action='store_true')

    parser.add_argument("--no-clean", action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    main()
