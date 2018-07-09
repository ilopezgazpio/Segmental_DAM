import os
import sys
import argparse
import numpy as np

def main(arguments):
    '''
    Read Multi NLI official dataset and produce txt files for sent1, sent2, label and pairID
    '''
    
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_folder', help="location of folder with the snli files")
    parser.add_argument('--out_folder', help="location of the output folder")
    
    args = parser.parse_args(arguments)

    for split in ["train", "dev", "test"]:        
        src_out = open(os.path.join(args.out_folder, "src-"+split+".txt"), "w")
        targ_out = open(os.path.join(args.out_folder, "targ-"+split+".txt"), "w")
        label_out = open(os.path.join(args.out_folder, "label-"+split+".txt"), "w")
        label_set = set(["neutral", "entailment", "contradiction"])

        if split == "test":
            pair_out = open(os.path.join(args.out_folder, "pairID-"+split+".txt"), "w")

        for line in open(os.path.join(args.data_folder, "multinli_1.0_"+split+"_matched.txt"),"r"):
            d = line.split("\t")
            label = d[0].strip()
            premise = " ".join(d[1].replace("(", "").replace(")", "").strip().split())
            hypothesis = " ".join(d[2].replace("(", "").replace(")", "").strip().split())

            if split == "test":
                pairID = d[8].strip()
                
            if label in label_set:
                src_out.write(premise + "\n")
                targ_out.write(hypothesis + "\n")
                label_out.write(label + "\n")

                if split == "test":
                    pair_out.write(pairID + "\n")

        src_out.close()
        targ_out.close()
        label_out.close()

        if split == "test":
            pair_out.close()

    
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
