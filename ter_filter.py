#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import math

from collections import namedtuple
from collections import Counter

import numpy
import scipy.spatial

RUN_TER_SCRIPT = "./eval/runTER.py"

Stats = namedtuple("Stats", "name min max avg std mode")


def main():
    global RUN_TER_SCRIPT
    args = parse_user_args()
    RUN_TER_SCRIPT = args.run_ter_script
    clean_files = []

    # Process source and target files
    src = str(args.src)
    trg = str(args.trg)

    if args.bpe:
        debug("Removing BPEs...")
        src = remove_bpe(src)
        trg = remove_bpe(trg)
        clean_files += [src, trg]

    table, header, score = run_ter(src, trg)
    stats = calculate_stats(table, header)
    debug("Initial TER: {:.4f}".format(score))
    if args.verbose:
        debug_stats(stats)

    if not args.ref_src or not args.ref_trg:
        debug("Exit: reference files not specified")
        if args.clean:
            for f in clean_files:
                os.remove(f)
        exit(0)

    # Process reference source and target files
    ref_src = str(args.ref_src)
    ref_trg = str(args.ref_trg)

    if args.shuffle:
        debug("Shuffling reference data")
        ref_src, ref_trg, tmp = shuffle_data(ref_src, ref_trg)
        clean_files += [ref_src, ref_trg, tmp]

    if args.reference_size:
        debug("Truncating reference data to {} sentences".format(
            args.reference_size))
        ref_src = subset_data(ref_src, n=args.reference_size)
        ref_trg = subset_data(ref_trg, n=args.reference_size)
        clean_files += [ref_src, ref_trg]

    ref_table, header, ref_score = run_ter(ref_src, ref_trg, clean=False)
    ref_stats = calculate_stats(ref_table, header)
    debug("Reference TER: {:.4f}".format(ref_score))
    if args.verbose:
        debug_stats(ref_stats)

    # Filter data
    idxs = find_outliers(table, ref_stats, args.tolerance)
    idxs += find_bad_neigbours(
        table, ref_table, args.neighbours, skip=set(idxs))

    rm_idxs = set(idxs)
    debug("Removing {} sentences ({} left)".format(
        len(idxs), len(table) - len(idxs)))

    # Evaluate filtered data
    if args.eval:
        new_src = remove_sentences(src, src + ".new", idxs)
        new_trg = remove_sentences(trg, trg + ".new", idxs)
        clean_files += [new_src, new_trg]

        new_table, _, new_score = run_ter(new_src, new_trg)
        new_stats = calculate_stats(new_table, header)
        debug("New TER: {:.4f}".format(new_score))
        if args.verbose:
            debug_stats(new_stats)

    # Create final output
    if args.output:
        count = wc(args.src)
        files = [open(f) for f in [args.trg, args.src] + args.add_to_outputs]
        debug("Writing output to {} from {} files".format(args.output, len(
            files)))
        with open(args.output, "w") as out_io:
            for i in range(count):
                line = "\t".join(f.next().strip() for f in files)
                if i not in idxs:
                    out_io.write(line + "\n")
        for file in files:
            file.close()

    if args.clean:
        for f in clean_files:
            os.remove(f)


def find_bad_neigbours(table1,
                       table2,
                       neighbours=10,
                       skip=set(),
                       columns=[0, 1, 2, 6, 7],
                       normalize=False):
    ter_factor = 1.0
    ter_tolerance = 0.95
    X1 = numpy.array(table1)
    X2 = numpy.array(table2)
    # print X1.shape, X2.shape

    for i in skip:
        table1[i] = [0.0] * len(table1[0])
    if columns:
        X1[:, 7] *= ter_factor
        X2[:, 7] *= ter_factor
        X1 = X1[:, columns]
        X2 = X2[:, columns]
    if normalize:
        X1 = X1 / X1.max(axis=0)
        X2 = X2 / X2.max(axis=0)
    sim_matrix = scipy.spatial.distance.cdist(X2, X1, metric='euclidean')

    # debug("  Min. distance: {}".format(numpy.amin(sim_matrix)))
    # debug("  Max. distance: {}".format(numpy.amax(sim_matrix)))
    # debug("  Mean distance: {}".format(numpy.mean(sim_matrix)))

    keep_idxs = set()
    count = Counter()
    max_count = 1
    for i, row in enumerate(sim_matrix):
        temp_idxs = []
        for j in row.argsort()[:50]:
            if len(temp_idxs) >= neighbours:
                break
            if count[j] > max_count:
                continue
            if j in skip:
                continue
            if X1[j, -1] > X2[i, -1] * ter_tolerance:
                # print X1[j,-1], X2[i,-1]
                continue
            temp_idxs.append(j)
            count.update([j])
        keep_idxs.update(temp_idxs)

    return [i for i in xrange(len(table1)) if i not in keep_idxs]


def find_outliers(table, ref_stats, tolerance=0.1):
    idxs = []
    for i, row in enumerate(table):
        for j, val in enumerate(row):
            if val > (1 + tolerance) * ref_stats[j].max:
                idxs.append(i)
    return idxs


def remove_sentences(in_file, out_file, idxs):
    out_io = open(out_file, "w")
    with open(in_file) as in_io:
        for i, line in enumerate(in_io):
            if i not in idxs:
                out_io.write(line)
    out_io.close()
    return out_file


def run_ter(source, target, clean=True):
    cmd = "python {} -s {} -r {} -i filter -d" \
        .format(RUN_TER_SCRIPT, source, target)
    if not clean:
        cmd += " --no-clean"
    debug("Running command:", cmd)
    output = os.popen(cmd).read().strip().split("\n")

    header = output[0].split()
    table = [[float(v) for v in line.split()] for line in output[1:-1]]
    score = float(output[-1].split()[-1])
    return table, header, score


def calculate_stats(table, header):
    rotated = zip(*table[::-1])
    stats = []
    for i, vals in enumerate(rotated):
        round_vals = [round(v * 2) / 2 for v in vals]
        mode = max(set(round_vals), key=vals.count)

        stat = Stats(
            name=header[i],
            min=min(vals),
            max=max(vals),
            avg=numpy.mean(vals),
            mode=mode,
            std=numpy.std(vals))
        stats.append(stat)
    return stats


def subset_data(file, n):
    os.popen("shuf {f} | head -n {n} > {f}.subs".format(f=file, n=n))
    return file + ".subs"


def shuffle_data(source, target):
    suffix = ".shuf.{}".format(os.getpid())
    shuf_file = source + "+trg" + suffix
    os.popen("paste {s} {t} | shuf > {f}".format(
        s=source, t=target, f=shuf_file))
    os.popen("cut -f1 {f} > {s}{e}".format(f=shuf_file, s=source, e=suffix))
    os.popen("cut -f2 {f} > {t}{e}".format(f=shuf_file, t=target, e=suffix))
    return source + suffix, target + suffix, shuf_file


def remove_bpe(file):
    os.popen("sed 's/@@ //g' {f} > {f}.nobpe".format(f=file))
    return file + ".nobpe"


def wc(file):
    return int(os.popen("wc -l {}".format(file)).read().strip().split()[0])


def debug(*args):
    print >> sys.stderr, " ".join(args)


def debug_stats(stats):
    debug("Statistics (min/max/avg/std/mode)")
    for stat in stats:
        vals = [str(round(v, 3)) for v in stat[1:]]
        debug("  {}\t: {}".format(stat.name, "\t".join(vals)))


def parse_user_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--src", metavar="FILE", help="input source file", required=True)
    parser.add_argument("--trg", metavar="FILE", help="input target file", required=True)
    parser.add_argument("--ref-src", metavar="FILE", help="reference source file")
    parser.add_argument("--ref-trg", metavar="FILE", help="reference target file")

    parser.add_argument("--output", metavar="FILE", help="output file with target-source tab-separated pair")
    parser.add_argument("--add-to-outputs", metavar="FILE", nargs="*", default=[], help="paste additional data to the output file")

    parser.add_argument("-r", "--reference-size", metavar="INT", type=int, help="the size of reference data to calculate statistics from")
    parser.add_argument("-s", "--shuffle", action='store_true', help="shuffle reference data")
    parser.add_argument("-t", "--tolerance", metavar="FLOAT", type=float, default=0.1, help="the tolerance for outlier values")
    parser.add_argument("-n", "--neighbours", metavar="INT", type=int, default=10, help="the number of neighbours for each reference sentence")
    parser.add_argument("--normalize", action='store_true', help="normalize TER features")

    parser.add_argument("--run-ter-script", metavar="PATH", default="./scripts/runTER.py", help="path to runTER.py script")
    parser.add_argument("--bpe", help="remove BPEs from source and target data", action='store_true')
    parser.add_argument("--eval", help="evaluate filtered data", action='store_true')
    parser.add_argument("--verbose", help="show statistic tables", action='store_true')
    parser.add_argument("--clean", help="remove temporary files", action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    main()
