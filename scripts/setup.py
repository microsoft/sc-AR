"""Setup file."""
import os
import DownloadData


def setup():
    """Download data and setup directories."""
    DownloadData.main()
    if not os.path.isdir('result'):
        os.makedirs('result')


setup()
