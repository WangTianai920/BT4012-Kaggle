"""Contract tests based on a bounded SoCLaaS draft, independently extended and checked."""
import os
import tempfile
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from final.submission_io import ROOT, validate_submission, write_submissions, resolve_data_dir, check_input_files

class SubmissionContract(unittest.TestCase):
    def setUp(self):
        self.sample=pd.DataFrame({'index':[11,22,33]})
    def frame(self,ids=(11,22,33),values=(.1,.5,.9)):
        return pd.DataFrame({'index':list(ids),'target':list(values)})
    def test_valid(self):
        self.assertTrue(validate_submission(self.frame(),self.sample,3)['checks_passed'])
    def test_invalid_outputs(self):
        cases=[self.frame(ids=(22,11,33)),self.frame(ids=(11,11,33)),
               self.frame(values=(.1,np.nan,.9)),self.frame(values=(.1,np.inf,.9)),
               self.frame(values=(-.1,.5,.9)),self.frame(values=(.1,.5,1.1)),
               self.frame(values=(.5,.5,.5)),self.frame(values=(.1,'bad',.9)),
               self.frame().rename(columns={'target':'prediction'}),self.frame().assign(extra=1)]
        for frame in cases:
            with self.subTest(frame=frame.shape), self.assertRaises(ValueError):
                validate_submission(frame,self.sample)
    def test_bad_template_or_count(self):
        for sample in [pd.DataFrame({'index':[11,11,33]}),pd.DataFrame({'index':[11,None,33]})]:
            with self.assertRaises(ValueError): validate_submission(self.frame(),sample)
        with self.assertRaises(ValueError): validate_submission(self.frame(),self.sample,2)
    def test_both_written_separately_and_references_optional(self):
        with tempfile.TemporaryDirectory() as tmp:
            results={'1019-6':np.array([.1,.5,.9]),'0929-1':np.array([.2,.6,.8])}
            summary=write_submissions(results,self.sample,tmp,expected_rows=3)
            self.assertEqual(set(summary),set(results))
            for name in results:
                self.assertTrue(summary[name]['checks_passed'])
                self.assertTrue((Path(tmp)/f'{name}.csv').exists())
            # A shifted optional reference must fail even if its format is valid.
            references=Path(tmp)/'references';references.mkdir()
            self.frame().to_csv(references/'1019-6.csv',index=False)
            with self.assertRaises(ValueError):
                write_submissions({'1019-6':np.array([.2,.5,.9])},self.sample,tmp,references,3)
    def test_validate_all_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                write_submissions({'1019-6':[.1,.5,.9],'0929-1':[.2,.2,.2]},self.sample,tmp,expected_rows=3)
            self.assertFalse(list(Path(tmp).glob('*.csv')))
    def test_relative_paths_and_explicit_override(self):
        with patch.dict(os.environ,{'BT4012_DATA_DIR':'some_data'}):
            self.assertEqual(resolve_data_dir(),ROOT/'some_data')
            self.assertEqual(resolve_data_dir('data'),(ROOT/'data').resolve())
    def test_final_feature_imports_do_not_require_xgboost(self):
        code = """import importlib.abc, sys
class BlockXGBoost(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'xgboost' or fullname.startswith('xgboost.'):
            raise ModuleNotFoundError('XGBoost intentionally blocked by portability test')
sys.meta_path.insert(0, BlockXGBoost())
from final.regenerate_features import regenerate
from final import pipeline
assert 'xgboost' not in sys.modules
"""
        subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def test_missing_input_message_lists_files(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(FileNotFoundError) as cm:
            check_input_files(tmp)
        self.assertIn('sample_submission.csv',str(cm.exception))

if __name__=='__main__':unittest.main()
