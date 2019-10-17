import os
import glob
import sys
import pickle

import numpy as np
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import v2c utils
sys.path.append(ROOT_DIR)  # To find local version of the library

# All used metrics
METRICS = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr"]

# From COCOEval code
class COCOScorer(object):
    def __init__(self):
        print('init COCO-EVAL scorer')
            
    def score(self, GT, RES, IDs, result_file):
        self.eval = {}
        self.imgToEval = {}
        gts = {}
        res = {}
        for ID in IDs:
#            print ID
            gts[ID] = GT[ID]
            res[ID] = RES[ID]
        print('tokenization...')
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]

#         result_file = '/home/anguyen/workspace/paper_src/2018.icra.v2c.source/output/' + net_id + '/prediction/score_result.txt'
        print('RESULT FILE: ', result_file)
        
        fwriter = open(result_file, 'w')
        
        # =================================================
        # Compute scores
        # =================================================
        eval = {}
        scores = []
        for scorer, method in scorers:
            print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, IDs, m)
                    print "%s: %0.3f"%(m, sc)
                    fwriter.write("%s %0.3f\n"%(m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, IDs, method)
                print("%s: %0.3f"%(method, score))
                fwriter.write("%s %0.3f\n"%(method, score))
                
        #for metric, score in self.eval.items():
        #    print '%s: %.3f'%(metric, score)
        for metric in METRICS:
            scores.append(self.eval[metric])

        return np.array(scores)
    
    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

def read_prediction_file(prediction_file):
    # Create dicts for ground truths and predictions
    gts_dict, pds_dict = {}, {}
    with open(prediction_file, 'r') as f:
        lines = prediction_file.read().split('\n')

        for i in range(0, len(lines) - 4, 4):
            id_line = lines[i+1]
            gt_line = lines[i+2]
            pd_line = lines[i+3]
            
            # Build individual ground truth dict
            curr_gt_dict = {}
            curr_gt_dict['image_id'] = id_line
            curr_gt_dict['cap_id'] = 0 # only 1 ground truth caption
            curr_gt_dict['caption'] = gt_line
            gts_dict[id_line] = [curr_gt_dict]
            
            # Build current individual prediction dict
            curr_pd_dict = {}
            curr_pd_dict['image_id'] = id_line
            curr_pd_dict['caption'] = pd_line
            pds_dict[id_line] = [curr_pd_dict]

    return gts_dict, pds_dict

def test_iit_v2c():
    # Get all generated predicted files
    prediction_files = sorted(glob.glob(os.path.join(ROOT_DIR, 'checkpoints', 'prediction', '*.txt')))
    
    scorer = COCOScorer()
    max_scores = np.zeros((len(METRICS), ))
    max_file = None
    for prediction_file in prediction_files:
        gts_dict, pds_dict = read_prediction_file(prediction_file)
        ids = list(gts_dict.keys())
        scores = scorer.score(grt_dic, prd_dic, ids, result_file)
        if scores > max_scores:
            max_scores = scores
            max_file = prediction_file

    print()
    print('Maximum Score with file', max_file)
    for i in range(len(max_scores)):
        print('%s: %0.3f' % (METRICS[i], max_scores[i]))

if __name__ == '__main__':
    main()