import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

if __name__ == '__main__':
    print("Finding and running all tests...")
    
    # Discover('tests', top_level_dir='.') stops Python from confusing tests/cbow with src/cbow!
    test_suite = unittest.defaultTestLoader.discover('tests', top_level_dir='.')
    
    # Run them with 'verbosity=2' so it prints the name of every test!
    unittest.TextTestRunner(verbosity=2).run(test_suite)