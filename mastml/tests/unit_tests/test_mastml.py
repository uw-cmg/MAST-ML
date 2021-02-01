import unittest
import shutil
import os
import sys
sys.path.insert(0, os.path.abspath('../../../'))

from mastml.mastml import Mastml

class TestMastml(unittest.TestCase):

    def test_mastml(self):
        mastml = Mastml(savepath='testdir')
        savepath = mastml.get_savepath
        metadata = mastml.get_mastml_metadata
        self.assertTrue(os.path.exists(os.path.join(savepath, 'mastml_metadata.json')))
        shutil.rmtree(savepath)
        return

if __name__=='__main__':
    unittest.main()